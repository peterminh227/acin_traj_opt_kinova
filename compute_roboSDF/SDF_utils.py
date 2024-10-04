import torch
import trimesh
import numpy as np
import glob
import os
import compute_roboSDF.utils_rdf as utils_rdf
import skimage
from mesh_to_sdf import mesh_to_voxels
import torch.nn.functional as F

CURDIR = "./"
class BPSDF():
    def __init__(self, n_func, domain_min, domain_max, robot, device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        self.robot = robot
        self.model_path = os.path.join(CURDIR, 'models')

    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self, t, use_derivative=False):
        # t is normalized to [0,1]
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (
                        1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def build_basis_function_from_points(self, p, use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self.build_bernstein_t(p, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func ** 2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func ** 3)
        if use_derivative == False:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]
            d_phi_x = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_y = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1D).view(-1, self.n_func ** 3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
            return phi_xyz, d_phi_xyz

    def train_bf_sdf(self, epoches=200):
        # represent SDF using basis functions
        mesh_path = os.path.join(CURDIR, "tesla_iiwa_basis/*.stl")
        mesh_files = glob.glob(mesh_path)
        mesh_files = sorted(mesh_files)  # [1:] #except finger
        mesh_dict = {}
        for i, mf in enumerate(mesh_files):
            mesh_name = mf.split('/')[-1].split('.')[0]
            mesh = trimesh.load(mf)
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices - offset, axis=1))
            mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
            mesh_dict[i] = {}
            mesh_dict[i]['mesh_name'] = mesh_name
            # load data
            data = np.load(f'./data/sdf_points/basis_iiwa_{mesh_name}.npy', allow_pickle=True).item()
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]
            wb = torch.zeros(self.n_func ** 3).float().to(self.device)
            B = (torch.eye(self.n_func ** 3) / 1e-4).float().to(self.device)
            # loss_list = []
            for iter in range(epoches):
                choice_near = np.random.choice(len(point_near_data), 1024, replace=False)
                p_near, sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(
                    self.device), torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)
                choice_random = np.random.choice(len(point_random_data), 256, replace=False)
                p_random, sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(
                    self.device), torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)
                p = torch.cat([p_near, p_random], dim=0)
                sdf = torch.cat([sdf_near, sdf_random], dim=0)
                phi_xyz, _ = self.build_basis_function_from_points(p.float().to(self.device), use_derivative=False)

                K = torch.matmul(B, phi_xyz.T).matmul(torch.linalg.inv(
                    (torch.eye(len(p)).float().to(self.device) + torch.matmul(torch.matmul(phi_xyz, B), phi_xyz.T))))
                B -= torch.matmul(K, phi_xyz).matmul(B)
                delta_wb = torch.matmul(K, (sdf - torch.matmul(phi_xyz, wb)).squeeze())
                # loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz,wb).squeeze(), sdf, reduction='mean').item()
                # loss_list.append(loss)
                wb += delta_wb
                offset_copy = np.copy(offset)
            print(f'mesh name {mesh_name} finished!')
            mesh_dict[i] = {
                'mesh_name': mesh_name,
                'weights': wb,
                'offset': torch.from_numpy(offset_copy),
                'scale': scale,

            }
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)
        torch.save(mesh_dict, f'{self.model_path}/BP_iiwa_{self.n_func}.pt')  # save the robot sdf model
        print(f'{self.model_path}/BP_iiwa_{self.n_func}.pt model saved!')

    def sdf_to_mesh(self, model, nbData, use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].to(self.device)

            domain = torch.linspace(self.domain_min, self.domain_max, nbData).to(self.device)
            grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
            p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d = []
            for p_s in p_split:
                phi_p, d_phi_p = self.build_basis_function_from_points(p_s, use_derivative)
                d_s = torch.matmul(phi_p, weights)
                d.append(d_s)
            d = torch.cat(d, dim=0)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0,
                spacing=np.array([(self.domain_max - self.domain_min) / nbData] * 3)
            )
            verts = verts - [1, 1, 1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list, mesh_name_list

    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        verts_list, faces_list, mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces, mesh_name in zip(verts_list, faces_list, mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts, faces)
            if vis:
                rec_mesh.show()
            if save_mesh_name != None:
                save_path = os.path.join(CURDIR, "output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh,
                                                    os.path.join(save_path, f"{save_mesh_name}_{mesh_name}.stl"))

    def get_whole_body_sdf_batch(self, x, pose, theta, model, use_derivative=True, used_links=[0, 1, 2, 3, 4, 5, 6, 7],
                                 return_index=False):

        B = len(theta)
        N = len(x)
        K = len(used_links)
        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()
        trans_list = self.robot.get_transformations_each_link(pose, theta)

        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(-1, 4,
                                                                                                       4)  # B,K,4,4
        x_robot_frame_batch = utils_rdf.transform_points(x.float(), torch.linalg.inv(fk_trans).float(),
                                                         device=self.device)  # B*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)  # B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=False)
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)  # K,B*N,-1
            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B, N).transpose(0, 1).reshape(B * K,
                                                                                                         N)  # B,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:
            phi, dphi = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)  # K,B*N,-1,4

            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)

            output = torch.einsum('ijkl,ik->ijl', phi_cat, weights_near).reshape(K, B, N, 4).transpose(0, 1).reshape(
                B * K, N, 4)
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * (scale.reshape(B, K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl', fk_rotation, gradient.transpose(1, 2)).transpose(1,
                                                                                                                2).reshape(
                B, K, N, 3)
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx_grad)[:, 0, :, :]
            # gradient_value = None
            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value

    def get_whole_body_sdf_with_joints_grad_batch(self, x, pose, theta, model, used_links=[0, 1, 2, 3, 4, 5, 6, 7]):

        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7,
                                                                                                7) * delta).reshape(B,
                                                                                                                    -1,
                                                                                                                    7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.unsqueeze(1).expand(B, 8, 4, 4).reshape(B * 8, 4, 4)
        sdf, _ = self.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=False, used_links=used_links)
        sdf = sdf.reshape(B, 8, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)

    def get_whole_body_normal_with_joints_grad_batch(self, x, pose, theta, model, used_links=[0, 1, 2, 3, 4, 5, 6, 7]):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7,
                                                                                                7) * delta).reshape(B,
                                                                                                                    -1,
                                                                                                                    7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.unsqueeze(1).expand(B, 8, 4, 4).reshape(B * 8, 4, 4)
        sdf, normal = self.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=True, used_links=used_links)
        normal = normal.reshape(B, 8, -1, 3).transpose(1, 2)
        return normal  # normal size: (B,N,8,3) normal[:,:,0,:] origin normal vector normal[:,:,1:,:] derivatives with respect to joints


class KUKAiiwaLayer(torch.nn.Module):
    # def __init__(self, device='cpu',mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"):
    def __init__(self, device='cpu', mesh_path="./tesla_iiwa_basis" + "/*.stl"):
        # The forward kinematics equations implemented here are     robot_mesh.show()from
        super().__init__()
        self.device = device
        self.mesh_path = mesh_path
        #self.meshes = self.load_meshes()

        # self.theta_min = [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895]
        # self.theta_max = [ 2.3093,  1.5133,  2.4937, -0.4461,  2.4800, 4.2094,  2.6895]

        self.theta_min = torch.tensor([-2.9670597283903604,
                                       -2.0943951023931953,
                                       -2.9670597283903604,
                                       -2.0943951023931953,
                                       -2.9670597283903604,
                                       -2.0943951023931953,
                                       -3.0543261909900767]).to(self.device)
        self.theta_max = torch.tensor([2.9670597283903604,
                                       2.0943951023931953,
                                       2.9670597283903604,
                                       2.0943951023931953,
                                       2.9670597283903604,
                                       2.0943951023931953,
                                       3.0543261909900767]).to(self.device)

        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.8 + self.theta_mid
        self.dof = len(self.theta_min)

        # meshes
        #self.link0 = self.meshes["base_link"][0]
        #self.link0_normals = self.meshes["base_link"][-1]
#
        #self.link1 = self.meshes["link_1"][0]
        #self.link1_normals = self.meshes["link_1"][-1]
#
        #self.link2 = self.meshes["link_2"][0]
        #self.link2_normals = self.meshes["link_2"][-1]
#
        #self.link3 = self.meshes["link_3"][0]
        #self.link3_normals = self.meshes["link_3"][-1]
#
        #self.link4 = self.meshes["link_4"][0]
        #self.link4_normals = self.meshes["link_4"][-1]
#
        #self.link5 = self.meshes["link_5"][0]
        #self.link5_normals = self.meshes["link_5"][-1]
#
        #self.link6 = self.meshes["link_6"][0]
        #self.link6_normals = self.meshes["link_6"][-1]
        #self.link7 = self.meshes["link_7"][0]
        #self.link7_normals = self.meshes["link_7"][-1]

        # mesh faces
        #self.robot_faces = [
        #    self.meshes["base_link"][1], self.meshes["link_1"][1], self.meshes["link_2"][1],
        #    self.meshes["link_3"][1], self.meshes["link_4"][1], self.meshes["link_5"][1],
        #    self.meshes["link_6"][1], self.meshes["link_7"][1]
        #]

        # self.vertices_face_areas = [
        #     self.meshes["link0"][2], self.meshes["link1"][2], self.meshes["link2"][2],
        #     self.meshes["link3"][2], self.meshes["link4"][2], self.meshes["link5"][2],
        #     self.meshes["link6"][2], self.meshes["link7"][2], self.meshes["link8"][2],
        #     self.meshes["finger"][2]
        # ]

        #self.num_vertices_per_part = [
        #    self.meshes["base_link"][0].shape[0], self.meshes["link_1"][0].shape[0], self.meshes["link_2"][0].shape[0],
        #    self.meshes["link_3"][0].shape[0], self.meshes["link_4"][0].shape[0], self.meshes["link_5"][0].shape[0],
        #    self.meshes["link_6"][0].shape[0], self.meshes["link_7"][0].shape[0]
        #]
        '''FK params'''
        self.trans_01 = torch.tensor([0, 0, 0.1575], dtype=torch.float32).to(self.device)  # joint location
        self.rotq_01 = torch.tensor([1, 0, 0, 0], dtype=torch.float32).to(self.device)
        self.trans_01_geom = torch.tensor([0, 0, -0.005], dtype=torch.float32).to(self.device)  # geom location
        self.rotq_01_geom = torch.tensor([1, 0, 0, 0], dtype=torch.float32).to(self.device)  # geom rotation
        self.axis_01 = [0, 0, 1]

        self.trans_12 = torch.tensor([0, 0, 0.20250000000000001], dtype=torch.float32).to(self.device)
        self.rotq_12 = torch.tensor([0, 0, 0.70710700000000004, 0.70710700000000004], dtype=torch.float32).to(
            self.device)
        self.trans_12_geom = torch.tensor([0, 0, -0.012999999999999999], dtype=torch.float32).to(
            self.device)  # geom location
        self.rotq_12_geom = torch.tensor([0, 0, 0, 1], dtype=torch.float32).to(self.device)  # geom rotation
        self.axis_12 = [0, 0, 1]

        # pos="0 0.23749999999999999 0" quat="0 0 0.70710700000000004 0.70710700000000004"
        self.trans_23 = torch.tensor([0, 0.23749999999999999, 0], dtype=torch.float32).to(self.device)
        self.rotq_23 = torch.tensor([0, 0, 0.70710700000000004, 0.70710700000000004], dtype=torch.float32).to(
            self.device)
        self.trans_23_geom = torch.tensor([0, 0, -0.0050000000000000001], dtype=torch.float32).to(self.device)
        self.rotq_23_geom = torch.tensor([1, 0, 0, 0], dtype=torch.float32).to(self.device)
        self.axis_23 = [0, 0, 1]

        # pos="0 0 0.1825" quat="0.70710700000000004 0.70710700000000004 0 0"
        self.trans_34 = torch.tensor([0, 0, 0.1825], dtype=torch.float32).to(self.device)
        self.rotq_34 = torch.tensor([0.70710700000000004, 0.70710700000000004, 0, 0], dtype=torch.float32).to(
            self.device)
        self.trans_34_geom = torch.tensor([0, 0, -0.010999999999999999], dtype=torch.float32).to(self.device)
        self.rotq_34_geom = torch.tensor([1, 0, 0, 0], dtype=torch.float32).to(self.device)
        self.axis_34 = [0, 0, 1]

        # pos="0 0.2175 0" quat="0 0 0.70710700000000004 0.70710700000000004"
        self.trans_45 = torch.tensor([0, 0.2175, 0], dtype=torch.float32).to(self.device)
        self.rotq_45 = torch.tensor([0, 0, 0.70710700000000004, 0.70710700000000004], dtype=torch.float32).to(
            self.device)
        self.trans_45_geom = torch.tensor([0, 0, -0.0050000000000000001], dtype=torch.float32).to(self.device)
        self.rotq_45_geom = torch.tensor([0, 0, 0, 1], dtype=torch.float32).to(self.device)
        self.axis_45 = [0, 0, 1]

        # pos="0 0 0.1825" quat="0.70710700000000004 0.70710700000000004 0 0"
        self.trans_56 = torch.tensor([0, 0, 0.1825], dtype=torch.float32).to(self.device)
        self.rotq_56 = torch.tensor([0.70710700000000004, 0.70710700000000004, 0, 0], dtype=torch.float32).to(
            self.device)
        self.trans_56_geom = torch.tensor([0, 0, -0.060999999999999999], dtype=torch.float32).to(self.device)
        self.rotq_56_geom = torch.tensor([0, 0, 0, 1], dtype=torch.float32).to(self.device)
        self.axis_56 = [0, 0, 1]

        # pos="0 0.081000000000000003 0" quat="0 0 0.70710700000000004 0.70710700000000004"
        self.trans_67 = torch.tensor([0, 0.081000000000000003, 0], dtype=torch.float32).to(self.device)
        self.rotq_67 = torch.tensor([0, 0, 0.70710700000000004, 0.70710700000000004], dtype=torch.float32).to(
            self.device)
        self.trans_67_geom = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)
        self.rotq_67_geom = torch.tensor([1, 0, 0, 0], dtype=torch.float32).to(self.device)
        self.axis_67 = [0, 0, 1]
        '''
        self.A0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A3 = torch.tensor(0.0825, dtype=torch.float32, device=device)
        self.A4 = torch.tensor(-0.0825, dtype=torch.float32, device=device)
        self.A5 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A6 = torch.tensor(0.088, dtype=torch.float32, device=device)
        self.A7 = torch.tensor(0.0, dtype=torch.float32, device=device)
        '''

    # def check_normal(self,verterices, normals):
    #     center = np.mean(verterices,axis=0)
    #     verts = torch.from_numpy(verterices-center).float()
    #     normals = torch.from_numpy(normals).float()
    #     cosine = torch.cosine_similarity(verts,normals).float()
    #     normals[cosine<0] = -normals[cosine<0]
    #     return normals

    def load_meshes(self):
        check_normal = False
        # mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}

        for mesh_file in mesh_files:
            if self.mesh_path.split('/')[-2] == 'visual':
                name = os.path.basename(mesh_file)[:-4].split('_')[0]
            else:
                name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            triangle_areas = trimesh.triangles.area(mesh.triangles)
            vert_area_weight = []
            # for i in range(mesh.vertices.shape[0]):
            #     vert_neighour_face = np.where(mesh.faces == i)[0]
            #     vert_area_weight.append(1000000*triangle_areas[vert_neighour_face].mean())
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            meshes[name] = [
                torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                # torch.LongTensor(np.asarray(mesh.faces)).to(self.device),
                mesh.faces,
                # torch.FloatTensor(np.asarray(vert_area_weight)).to(self.device),
                # vert_area_weight,
                # torch.FloatTensor(mesh.vertex_normals)
                torch.cat((torch.FloatTensor(np.array(mesh.vertex_normals)), temp), dim=-1).to(self.device).to(
                    torch.float),
            ]
        return meshes

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert a quaternion to a rotation matrix.
        quaternion should be of shape [batch_size, 4].
        """
        q = F.normalize(quaternion, dim=-1)  # Ensure the quaternion is normalized
        w, x, y, z = q.unbind(-1)

        # Compute the rotation matrix
        b = quaternion.size(0)
        rot_matrix = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        ], dim=-1).view(b, 3, 3)

        return rot_matrix

    def create_rotation_matrix(self, axis, theta):
        """
        Create a rotation matrix for rotation around a specified axis.
        axis should be a tensor [3] indicating the axis (e.g., [1, 0, 0] for x).
        theta should be of shape [batch_size, 1].
        """
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        b = theta.size(0)

        if axis[0] == 1:  # Rotation around the x-axis
            '''
            rot_matrix = torch.stack([
                torch.ones(b, device=theta.device), torch.zeros(b, device=theta.device), torch.zeros(b, device=theta.device),
                torch.zeros(b, device=theta.device), cos_theta, -sin_theta,
                torch.zeros(b, device=theta.device), sin_theta, cos_theta
            ], dim=-1).view(b, 3, 3)
            '''
            rot_matrix = torch.cat([
                torch.ones_like(cos_theta), torch.zeros_like(cos_theta), torch.zeros_like(cos_theta),
                torch.zeros_like(cos_theta), cos_theta, -sin_theta,
                torch.zeros_like(cos_theta), sin_theta, cos_theta
            ], dim=1).view(b, 3, 3)
        elif axis[1] == 1:  # Rotation around the y-axis
            '''
            rot_matrix = torch.stack([
                cos_theta, torch.zeros(b, device=theta.device), sin_theta,
                torch.zeros(b, device=theta.device), torch.ones(b, device=theta.device), torch.zeros(b, device=theta.device),
                -sin_theta, torch.zeros(b, device=theta.device), cos_theta
            ], dim=-1).view(b, 3, 3)
            '''
            rot_matrix = torch.cat([
                cos_theta, torch.zeros_like(cos_theta), sin_theta,
                torch.zeros_like(cos_theta), torch.ones_like(cos_theta), torch.zeros_like(cos_theta),
                -sin_theta, torch.zeros_like(cos_theta), cos_theta
            ], dim=1).view(b, 3, 3)

        elif axis[2] == 1:  # Rotation around the z-axis
            '''
            rot_matrix = torch.stack([
                cos_theta, -sin_theta, torch.zeros(b, device=theta.device),
                sin_theta, cos_theta, torch.zeros(b, device=theta.device),
                torch.zeros(b, device=theta.device), torch.zeros(b, device=theta.device), torch.ones(b, device=theta.device)
            ], dim=-1).view(b, 3, 3)
            '''
            rot_matrix = torch.cat([
                cos_theta, -sin_theta, torch.zeros_like(cos_theta),
                sin_theta, cos_theta, torch.zeros_like(cos_theta),
                torch.zeros_like(cos_theta), torch.zeros_like(cos_theta), torch.ones_like(cos_theta)
            ], dim=1).view(b, 3, 3)
        else:
            raise ValueError("Axis must be one of [1, 0, 0], [0, 1, 0], or [0, 0, 1].")

        return rot_matrix

    def compute_iiwa_FK(self, batch_size, trans, rotq, theta, axis):
        """
        Compute the overall homogeneous transformation matrix.
        """
        # Ensure trans and rotq are expanded to match batch size
        trans = trans.expand(batch_size, 3)
        rotq = rotq.expand(batch_size, 4)

        # Convert quaternion to rotation matrix
        rotm = self.quaternion_to_rotation_matrix(rotq)

        # Create the homogeneous transformation matrix for initial rotation and translation
        homogeneous_matrix_origin = torch.eye(4, device=trans.device).expand(batch_size, 4, 4).clone()
        homogeneous_matrix_origin[:, :3, :3] = rotm
        homogeneous_matrix_origin[:, :3, 3] = trans

        # Create rotation matrix for the specified axis
        rotm_joint = self.create_rotation_matrix(axis, theta)

        # Create the homogeneous transformation matrix for the axis rotation
        homogeneous_matrix_joint = torch.eye(4, device=theta.device).expand(batch_size, 4, 4).clone()
        homogeneous_matrix_joint[:, :3, :3] = rotm_joint

        # Combine the two transformations
        Homo = torch.bmm(homogeneous_matrix_origin, homogeneous_matrix_joint)
        # Homo = torch.bmm(homogeneous_matrix_joint,homogeneous_matrix_origin)
        return Homo

    def compute_iiwa_FK_geom(self, batch_size, trans, rotq, trans_geom, rotq_geom, theta, axis):
        """
        Compute the overall homogeneous transformation matrix.
        """
        # Ensure trans and rotq are expanded to match batch size
        trans = trans.expand(batch_size, 3)
        rotq = rotq.expand(batch_size, 4)
        # Convert quaternion to rotation matrix
        rotm = self.quaternion_to_rotation_matrix(rotq)

        # Create the homogeneous transformation matrix for initial rotation and translation
        homogeneous_matrix_origin = torch.eye(4, device=trans.device).expand(batch_size, 4, 4).clone()
        homogeneous_matrix_origin[:, :3, :3] = rotm
        homogeneous_matrix_origin[:, :3, 3] = trans
        '''GEOM'''
        trans_geom = trans_geom.expand(batch_size, 3)
        rotq_geom = rotq_geom.expand(batch_size, 4)
        # Convert quaternion to rotation matrix
        rotm_geom = self.quaternion_to_rotation_matrix(rotq_geom)
        # Create the homogeneous transformation matrix for initial rotation and translation
        homogeneous_matrix_origin_geom = torch.eye(4, device=trans.device).expand(batch_size, 4, 4).clone()
        homogeneous_matrix_origin_geom[:, :3, :3] = rotm_geom
        homogeneous_matrix_origin_geom[:, :3, 3] = trans_geom

        # Create rotation matrix for the specified axis
        rotm_joint = self.create_rotation_matrix(axis, theta)

        # Create the homogeneous transformation matrix for the axis rotation
        homogeneous_matrix_joint = torch.eye(4, device=theta.device).expand(batch_size, 4, 4).clone()
        homogeneous_matrix_joint[:, :3, :3] = rotm_joint

        # Combine the two transformations
        Homo = torch.bmm(homogeneous_matrix_origin, homogeneous_matrix_origin_geom)
        Homo = torch.bmm(Homo, homogeneous_matrix_joint)

        # Homo = torch.bmm(homogeneous_matrix_joint,homogeneous_matrix_origin)
        return Homo

    def forward(self, pose, theta):
        batch_size = theta.shape[0]
        link0_vertices = self.link0.repeat(batch_size, 1, 1)
        # print(link0_vertices.shape)
        link0_vertices = torch.matmul(pose,
                                      link0_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link0_normals = self.link0_normals.repeat(batch_size, 1, 1)
        link0_normals = torch.matmul(pose,
                                     link0_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        link1_vertices = self.link1.repeat(batch_size, 1, 1)
        T01 = self.compute_iiwa_FK(batch_size, self.trans_01, self.rotq_01, theta[:, 0].view(batch_size, -1),
                                   self.axis_01)

        T01_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_01, self.rotq_01,
                                             self.trans_01_geom, self.rotq_01_geom,
                                             theta[:, 0].view(batch_size, -1), self.axis_01)
        # T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
        #                              0.333, theta[:, 0], batch_size).float()
        # T01 = self.comp_T_01(batch_size,theta[:,0])

        link2_vertices = self.link2.repeat(batch_size, 1, 1)
        T12 = self.compute_iiwa_FK(batch_size, self.trans_12, self.rotq_12, theta[:, 1].view(batch_size, -1),
                                   self.axis_12)
        T12_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_12, self.rotq_12,
                                             self.trans_12_geom, self.rotq_12_geom,
                                             theta[:, 1].view(batch_size, -1), self.axis_12)

        # T12 = self.forward_kinematics(self.A1, torch.tensor(-np.pi/2., dtype=torch.float32, device=self.device),
        #                              0, theta[:, 1], batch_size).float()
        link3_vertices = self.link3.repeat(batch_size, 1, 1)
        T23_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_23, self.rotq_23,
                                             self.trans_23_geom, self.rotq_23_geom,
                                             theta[:, 2].view(batch_size, -1), self.axis_23)
        T23 = self.compute_iiwa_FK(batch_size, self.trans_23, self.rotq_23, theta[:, 2].view(batch_size, -1),
                                   self.axis_23)

        # T23 = self.forward_kinematics(self.A2, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
        #                              0.316, theta[:, 2], batch_size).float()
        link4_vertices = self.link4.repeat(batch_size, 1, 1)
        T34_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_34, self.rotq_34,
                                             self.trans_34_geom, self.rotq_34_geom,
                                             theta[:, 3].view(batch_size, -1), self.axis_34)
        T34 = self.compute_iiwa_FK(batch_size, self.trans_34, self.rotq_34, theta[:, 3].view(batch_size, -1),
                                   self.axis_34)
        # T34 = self.forward_kinematics(self.A3, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
        #                              0, theta[:, 3], batch_size).float()
        link5_vertices = self.link5.repeat(batch_size, 1, 1)
        T45_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_45, self.rotq_45,
                                             self.trans_45_geom, self.rotq_45_geom,
                                             theta[:, 4].view(batch_size, -1), self.axis_45)
        T45 = self.compute_iiwa_FK(batch_size, self.trans_45, self.rotq_45, theta[:, 4].view(batch_size, -1),
                                   self.axis_45)
        # T45 = self.forward_kinematics(self.A4, torch.tensor(-np.pi/2., dtype=torch.float32, device=self.device),
        #                              0.384, theta[:, 4], batch_size).float()
        link6_vertices = self.link6.repeat(batch_size, 1, 1)
        T56_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_56, self.rotq_56,
                                             self.trans_56_geom, self.rotq_56_geom,
                                             theta[:, 5].view(batch_size, -1), self.axis_56)
        T56 = self.compute_iiwa_FK(batch_size, self.trans_56, self.rotq_56, theta[:, 5].view(batch_size, -1),
                                   self.axis_56)
        # T56 = self.forward_kinematics(self.A5, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
        #                              0, theta[:, 5], batch_size).float()
        link7_vertices = self.link7.repeat(batch_size, 1, 1)
        T67_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_67, self.rotq_67,
                                             self.trans_67_geom, self.rotq_67_geom,
                                             theta[:, 6].view(batch_size, -1), self.axis_67)
        T67 = self.compute_iiwa_FK(batch_size, self.trans_67, self.rotq_67, theta[:, 6].view(batch_size, -1),
                                   self.axis_67)
        # T67 = self.forward_kinematics(self.A6, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
        #                              0, theta[:, 6], batch_size).float()
        # link8_vertices = self.link8.repeat(batch_size, 1, 1)
        # T78 = self.forward_kinematics(self.A7, torch.tensor(0, dtype=torch.float32, device=self.device),
        #                              0.107, -np.pi/4*torch.ones_like(theta[:,0],device=self.device), batch_size).float()
        # finger_vertices = self.finger.repeat(batch_size, 1, 1)

        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        pose_to_T01_geom = torch.matmul(pose_to_Tw0, T01_geom)

        link1_vertices = torch.matmul(
            pose_to_T01_geom,
            link1_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link1_normals = self.link1_normals.repeat(batch_size, 1, 1)
        link1_normals = torch.matmul(pose_to_T01_geom,
                                     link1_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        pose_to_T12_geom = torch.matmul(pose_to_T01, T12_geom)
        link2_vertices = torch.matmul(
            pose_to_T12_geom,
            link2_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link2_normals = self.link2_normals.repeat(batch_size, 1, 1)
        link2_normals = torch.matmul(pose_to_T12_geom,
                                     link2_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        pose_to_T23_geom = torch.matmul(pose_to_T12, T23_geom)
        link3_vertices = torch.matmul(
            pose_to_T23_geom,
            link3_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link3_normals = self.link3_normals.repeat(batch_size, 1, 1)
        link3_normals = torch.matmul(pose_to_T23_geom,
                                     link3_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        pose_to_T34_geom = torch.matmul(pose_to_T23, T34_geom)
        link4_vertices = torch.matmul(
            pose_to_T34_geom,
            link4_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link4_normals = self.link4_normals.repeat(batch_size, 1, 1)
        link4_normals = torch.matmul(pose_to_T34_geom,
                                     link4_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        pose_to_T45_geom = torch.matmul(pose_to_T34, T45_geom)
        link5_vertices = torch.matmul(
            pose_to_T45_geom,
            link5_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link5_normals = self.link5_normals.repeat(batch_size, 1, 1)
        link5_normals = torch.matmul(pose_to_T45_geom,
                                     link5_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T56 = torch.matmul(pose_to_T45, T56)
        pose_to_T56_geom = torch.matmul(pose_to_T45, T56_geom)

        link6_vertices = torch.matmul(
            pose_to_T56_geom,
            link6_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link6_normals = self.link6_normals.repeat(batch_size, 1, 1)
        link6_normals = torch.matmul(pose_to_T56_geom,
                                     link6_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T67 = torch.matmul(pose_to_T56, T67)
        pose_to_T67_geom = torch.matmul(pose_to_T56, T67_geom)
        link7_vertices = torch.matmul(
            pose_to_T67_geom,
            link7_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link7_normals = self.link7_normals.repeat(batch_size, 1, 1)
        link7_normals = torch.matmul(pose_to_T67_geom,
                                     link7_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return [link0_vertices, link1_vertices, link2_vertices, \
                link3_vertices, link4_vertices, link5_vertices, \
                link6_vertices, link7_vertices, \
                link0_normals, link1_normals, link2_normals, \
                link3_normals, link4_normals, link5_normals, \
                link6_normals, link7_normals]

    def get_transformations_each_link(self, pose, theta):
        batch_size = theta.shape[0]

        T01 = self.compute_iiwa_FK(batch_size, self.trans_01, self.rotq_01, theta[:, 0].view(batch_size, -1),
                                   self.axis_01)
        T01_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_01, self.rotq_01,
                                             self.trans_01_geom, self.rotq_01_geom,
                                             theta[:, 0].view(batch_size, -1), self.axis_01)

        # link2_vertices = self.link2.repeat(batch_size, 1, 1)
        T12 = self.compute_iiwa_FK(batch_size, self.trans_12, self.rotq_12, theta[:, 1].view(batch_size, -1),
                                   self.axis_12)
        T12_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_12, self.rotq_12,
                                             self.trans_12_geom, self.rotq_12_geom,
                                             theta[:, 1].view(batch_size, -1), self.axis_12)
        # link3_vertices = self.link3.repeat(batch_size, 1, 1)
        T23 = self.compute_iiwa_FK(batch_size, self.trans_23, self.rotq_23, theta[:, 2].view(batch_size, -1),
                                   self.axis_23)
        T23_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_23, self.rotq_23,
                                             self.trans_23_geom, self.rotq_23_geom,
                                             theta[:, 2].view(batch_size, -1), self.axis_23)
        # link4_vertices = self.link4.repeat(batch_size, 1, 1)
        T34 = self.compute_iiwa_FK(batch_size, self.trans_34, self.rotq_34, theta[:, 3].view(batch_size, -1),
                                   self.axis_34)
        T34_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_34, self.rotq_34,
                                             self.trans_34_geom, self.rotq_34_geom,
                                             theta[:, 3].view(batch_size, -1), self.axis_34)
        T45 = self.compute_iiwa_FK(batch_size, self.trans_45, self.rotq_45, theta[:, 4].view(batch_size, -1),
                                   self.axis_45)
        T45_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_45, self.rotq_45,
                                             self.trans_45_geom, self.rotq_45_geom,
                                             theta[:, 4].view(batch_size, -1), self.axis_45)
        T56 = self.compute_iiwa_FK(batch_size, self.trans_56, self.rotq_56, theta[:, 5].view(batch_size, -1),
                                   self.axis_56)
        T56_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_56, self.rotq_56,
                                             self.trans_56_geom, self.rotq_56_geom,
                                             theta[:, 5].view(batch_size, -1), self.axis_56)
        T67 = self.compute_iiwa_FK(batch_size, self.trans_67, self.rotq_67, theta[:, 6].view(batch_size, -1),
                                   self.axis_67)
        T67_geom = self.compute_iiwa_FK_geom(batch_size,
                                             self.trans_67, self.rotq_67,
                                             self.trans_67_geom, self.rotq_67_geom,
                                             theta[:, 6].view(batch_size, -1), self.axis_67)
        # finger_vertices = self.finger.repeat(batch_size, 1, 1)
        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        pose_to_T56 = torch.matmul(pose_to_T45, T56)
        pose_to_T67 = torch.matmul(pose_to_T56, T67)

        '''GEOM'''
        # pose_to_Tw0 = pose
        pose_to_T01_geom = torch.matmul(pose_to_Tw0, T01_geom)
        pose_to_T12_geom = torch.matmul(pose_to_T01, T12_geom)
        pose_to_T23_geom = torch.matmul(pose_to_T12, T23_geom)
        pose_to_T34_geom = torch.matmul(pose_to_T23, T34_geom)
        pose_to_T45_geom = torch.matmul(pose_to_T34, T45_geom)
        pose_to_T56_geom = torch.matmul(pose_to_T45, T56_geom)
        pose_to_T67_geom = torch.matmul(pose_to_T56, T67_geom)
        # return [pose_to_Tw0,pose_to_T01,pose_to_T12,pose_to_T23,pose_to_T34,pose_to_T45,pose_to_T56,pose_to_T67]
        return [pose_to_Tw0, pose_to_T01_geom, pose_to_T12_geom, pose_to_T23_geom, pose_to_T34_geom, pose_to_T45_geom,
                pose_to_T56_geom, pose_to_T67_geom]

    def get_eef(self, pose, theta, link=-1):
        poses = self.get_transformations_each_link(pose, theta)
        pos = poses[link][:, :3, 3]
        rot = poses[link][:, :3, :3]
        return pos, rot

    def get_robot_mesh(self, vertices_list, faces):

        link0_verts = vertices_list[0]
        link0_faces = faces[0]

        link1_verts = vertices_list[1]
        link1_faces = faces[1]

        link2_verts = vertices_list[2]
        link2_faces = faces[2]

        link3_verts = vertices_list[3]
        link3_faces = faces[3]

        link4_verts = vertices_list[4]
        link4_faces = faces[4]

        link5_verts = vertices_list[5]
        link5_faces = faces[5]

        link6_verts = vertices_list[6]
        link6_faces = faces[6]

        link7_verts = vertices_list[7]
        link7_faces = faces[7]
        link0_mesh = trimesh.Trimesh(link0_verts, link0_faces)
        # link0_mesh.visual.face_colors = [150,150,150]
        link1_mesh = trimesh.Trimesh(link1_verts, link1_faces)
        # link1_mesh.visual.face_colors = [150,150,150]
        link2_mesh = trimesh.Trimesh(link2_verts, link2_faces)
        # link2_mesh.visual.face_colors = [150,150,150]
        link3_mesh = trimesh.Trimesh(link3_verts, link3_faces)
        # link3_mesh.visual.face_colors = [150,150,150]
        link4_mesh = trimesh.Trimesh(link4_verts, link4_faces)
        # link4_mesh.visual.face_colors = [150,150,150]
        link5_mesh = trimesh.Trimesh(link5_verts, link5_faces)
        # link5_mesh.visual.face_colors = [250,150,150]
        link6_mesh = trimesh.Trimesh(link6_verts, link6_faces)
        # link6_mesh.visual.face_colors = [250,150,150]
        link7_mesh = trimesh.Trimesh(link7_verts, link7_faces)
        # link7_mesh.visual.face_colors = [250,150,150]
        # link8_mesh = trimesh.Trimesh(link8_verts, link8_faces)
        # link8_mesh.visual.face_colors = [250,150,150]

        robot_mesh = [
            link0_mesh,
            link1_mesh,
            link2_mesh,
            link3_mesh,
            link4_mesh,
            link5_mesh,
            link6_mesh,
            link7_mesh,
            # link8_mesh
        ]
        # robot_mesh = np.sum(robot_mesh)
        return robot_mesh

    def get_forward_robot_mesh(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)
        vertices_list = [[
            outputs[0][i].detach().cpu().numpy(),
            outputs[1][i].detach().cpu().numpy(),
            outputs[2][i].detach().cpu().numpy(),
            outputs[3][i].detach().cpu().numpy(),
            outputs[4][i].detach().cpu().numpy(),
            outputs[5][i].detach().cpu().numpy(),
            outputs[6][i].detach().cpu().numpy(),
            outputs[7][i].detach().cpu().numpy()] for i in range(batch_size)]

        mesh = [self.get_robot_mesh(vertices, self.robot_faces) for vertices in vertices_list]
        return mesh

    def get_forward_vertices(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)
        robot_vertices = torch.cat((
            outputs[0].view(batch_size, -1, 3),
            outputs[1].view(batch_size, -1, 3),
            outputs[2].view(batch_size, -1, 3),
            outputs[3].view(batch_size, -1, 3),
            outputs[4].view(batch_size, -1, 3),
            outputs[5].view(batch_size, -1, 3),
            outputs[6].view(batch_size, -1, 3),
            outputs[7].view(batch_size, -1, 3)), 1)  # .squeeze()

        robot_vertices_normal = torch.cat((
            outputs[8].view(batch_size, -1, 3),
            outputs[9].view(batch_size, -1, 3),
            outputs[10].view(batch_size, -1, 3),
            outputs[11].view(batch_size, -1, 3),
            outputs[12].view(batch_size, -1, 3),
            outputs[13].view(batch_size, -1, 3),
            outputs[14].view(batch_size, -1, 3),
            outputs[15].view(batch_size, -1, 3)), 1)  # .squeeze()
        return robot_vertices, robot_vertices_normal

