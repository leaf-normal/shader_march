import os
import re
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from collections import defaultdict
import numpy as np
from PIL import Image

class MTLParser:
    """Parse MTL file and extract material parameters"""
    def __init__(self, mtl_path, output_dir=None):
        self.materials = {}
        self.current_material = None
        self.output_dir = output_dir
        
        if not os.path.exists(mtl_path):
            print("Warning: MTL file not found: {}".format(mtl_path))
            return
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            command = parts[0].lower()
            
            if command == 'newmtl':
                self.current_material = parts[1]
                self.materials[self.current_material] = {}
            elif self.current_material and command in ['ka', 'kd', 'ks', 'ke', 'tf']:
                if len(parts) >= 4:
                    values = [float(x) for x in parts[1:4]]
                    self.materials[self.current_material][command] = values
            elif self.current_material and command in ['d', 'tr', 'ns', 'ni', 'illum']:
                if len(parts) >= 2:
                    value = float(parts[1])
                    self.materials[self.current_material][command] = value
            elif self.current_material and command == 'map_kd':
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_kd'] = filename
            elif self.current_material and (command == 'map_bump' or command == 'bump'):
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_bump'] = filename
                    i = 1
                    while i < len(parts) - 1:
                        if parts[i] == '-bm' and i + 1 < len(parts):
                            try:
                                bump_multiplier = float(parts[i + 1])
                                self.materials[self.current_material]['bump_multiplier'] = bump_multiplier
                                print(f"  Found bump_multiplier for {self.current_material}: {bump_multiplier}")
                            except ValueError:
                                pass
                            break
                        i += 1
            elif self.current_material and command == 'map_roughness':
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_roughness'] = filename
            elif self.current_material and (command == 'map_pr' or command == 'map_rough'):
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_roughness'] = filename
            elif self.current_material and command == 'map_ns':
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_ns'] = filename
            elif self.current_material and command == 'map_metallic':
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_metallic'] = filename
            elif self.current_material and (command == 'map_d' or command == 'map_tr'):
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_' + command] = filename
            elif self.current_material and command == 'map_ke':
                filename = self._get_texture_filename(parts[1:])
                if filename:
                    self.materials[self.current_material]['map_ke'] = filename
            else:
                if self.current_material and command.startswith('map_'):
                    filename = self._get_texture_filename(parts[1:])
                    if filename:
                        self.materials[self.current_material][command] = filename
                        print(f"[Unknown texture] {self.current_material}: {command} = {filename}")
    
    def _get_texture_filename(self, args):
        """Extract texture filename from arguments, skipping options"""
        if not args:
            return None
        
        i = 0
        filename = None
        
        while i < len(args):
            arg = args[i]
            
            if arg.startswith('-'):
                if arg in ['-bm', '-blendu', '-blendv', '-boost', '-imfchan', '-texres']:
                    i += 2
                elif arg == '-cc':
                    i += 2
                elif arg in ['-mm', '-o', '-s', '-t']:
                    i += 4
                elif arg == '-clamp':
                    i += 1
                else:
                    i += 1
            else:
                filename = arg
                i += 1
        
        return filename
    

    def _read_dds_file(self, full_path):
        """Read DDS file"""
        try:
            from PIL import Image
            img = Image.open(full_path)
            data = np.array(img, dtype=np.float32)
            
            if len(data.shape) == 3 and data.shape[2] == 3:
                pass
            elif len(data.shape) == 2:
                data = np.stack([data] * 3, axis=-1)
            
            if data.dtype == np.uint8:
                data = data / 255.0
            
            print(f"  Loaded DDS: {full_path}, shape={data.shape}")
            return data
        except ImportError:
            print("  Error: PIL is required for DDS support")
            return None
        except Exception as e:
            print(f"  Warning: Failed to read DDS: {e}")
            return None

    def _copy_texture_to_output(self, texture_path, base_dir, new_filename=None):
        """
        Copy texture file to output directory.
        Returns the new filename or original filename.
        """
        if not texture_path or not self.output_dir:
            return texture_path
        
        full_path = os.path.join(base_dir, texture_path)
        if not os.path.exists(full_path):
            print(f"  Warning: Texture file not found, skipping copy: {full_path}")
            return texture_path
        
        output_filename = new_filename if new_filename else os.path.basename(texture_path)
        output_path = os.path.join(self.output_dir, output_filename)
        
        if os.path.exists(output_path) and os.path.samefile(full_path, output_path):
            return output_filename
        try:
            import shutil
            shutil.copy2(full_path, output_path)
            print(f"  Copied texture to output: {texture_path} -> {output_filename}")
            return output_filename
        except Exception as e:
            print(f"  Warning: Failed to copy texture {full_path}: {e}")
            return texture_path


    def _apply_bump_multiplier(self, normal_data, multiplier):
        """
        Bake bump coefficient into normal map.
        Args:
            normal_data: numpy array with shape (H, W, 3), range [0,1]
            multiplier: float, bump intensity coefficient
        """
        normal = normal_data * 2.0 - 1.0
        
        normal[:, :, 2] /= multiplier
        
        norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
        normal = normal / (norm + 1e-6)
        
        normal = normal * 0.5 + 0.5
        
        return np.clip(normal, 0.0, 1.0)
    
    def _read_exr_file(self, full_path):
        """读取EXR文件，优先使用OpenCV，回退到imageio"""
        try:
            import cv2
            data = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if data is not None:
                data = data.astype(np.float32)
                if len(data.shape) == 3 and data.shape[2] == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                print(f"  Loaded EXR via OpenCV: {full_path}, shape={data.shape}")
                return data
        except ImportError:
            pass
        except Exception as e:
            print(f"  OpenCV read failed, trying imageio...: {e}")

        try:
            import imageio
            data = imageio.imread(full_path)
            data = np.clip(data, 0.0, 1.0).astype(np.float32)
            print(f"  Loaded EXR via imageio: {full_path}, shape={data.shape}")
            return data
        except ImportError:
            print("  Error: Neither OpenCV nor imageio is installed for EXR support")
            print("  Please install: pip install opencv-python imageio")
            return None
        except Exception as e:
            print(f"  Warning: imageio failed to read EXR: {e}")
            return None
    
    def _load_texture(self, texture_path, base_dir):
        """加载纹理图像，支持DDS/EXR转换，返回灰度numpy数组"""
        full_path = os.path.join(base_dir, texture_path)
        if not os.path.exists(full_path):
            print(f"  Warning: Texture not found: {full_path}")
            return None
        
        try:
            if texture_path.lower().endswith('.dds'):
                data = self._read_dds_file(full_path)
                if data is None:
                    return None
                
                if len(data.shape) == 3:
                    gray_data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
                else:
                    gray_data = data
                
                return np.clip(gray_data, 0.0, 1.0)
            elif texture_path.lower().endswith('.exr'):
                data = self._read_exr_file(full_path)
                if data is None:
                    return None
                
                if len(data.shape) == 3:
                    gray_data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
                else:
                    gray_data = data
                
                return np.clip(gray_data, 0.0, 1.0)
            else:
                img = Image.open(full_path)
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Warning: Failed to load texture {full_path}: {e}")
            return None

    def _save_converted_dds_to_png(self, channel_name, dds_path, data, mat_name, target_size):
        """保存转换后的DDS为单独的PNG文件"""
        clean_name = re.sub(r'[^\w]', '_', mat_name)[:30]
        output_filename = f"{clean_name}_{channel_name}.png"
        
        if self.output_dir:
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            output_path = output_filename
        
        if data.shape != target_size:
            img_pil = Image.fromarray((data * 255).clip(0, 255).astype(np.uint8))
            img_pil = img_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
            data = np.array(img_pil, dtype=np.float32) / 255.0
        
        img_uint8 = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'L') 
        img.save(output_path, 'PNG')
        print(f"  Saved converted DDS ({channel_name}) to: {output_path}")

    def _check_texture_exists(self, texture_path, base_dir):
        """检查纹理文件是否存在"""
        if not texture_path:
            return False
        full_path = os.path.join(base_dir, texture_path)
        exists = os.path.exists(full_path)
        if not exists:
            print(f"  Warning: Texture file not found: {texture_path}")
        return exists


    def _load_and_process_normal_map(self, normal_path, base_dir, bump_multiplier=1.0):
        """加载并处理法线贴图（支持DDS/EXR转换和bump系数烘焙）"""
        full_path = os.path.join(base_dir, normal_path)
        if not os.path.exists(full_path):
            print(f"  Warning: Normal map not found: {full_path}")
            return None
        
        try:
            if normal_path.lower().endswith('.dds'):
                data = self._read_dds_file(full_path)
                if data is None:
                    return None
                normal_data = data
            elif normal_path.lower().endswith('.exr'):
                data = self._read_exr_file(full_path)
                if data is None:
                    return None
                normal_data = data
            else:
                img = Image.open(full_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                normal_data = np.array(img, dtype=np.float32) / 255.0
                print(f"  Loaded normal map: {normal_path}")
            
            if len(normal_data.shape) == 2:
                normal_data = np.stack([normal_data] * 3, axis=-1)
                print(f"  Converted single channel to RGB")
            elif len(normal_data.shape) == 3:
                if normal_data.shape[2] == 1:
                    normal_data = np.repeat(normal_data, 3, axis=2)
                elif normal_data.shape[2] == 4:
                    normal_data = normal_data[:, :, :3]
            
            if bump_multiplier != 1.0:
                normal_data = self._apply_bump_multiplier(normal_data, bump_multiplier)
                print(f"  Applied bump multiplier: {bump_multiplier}")
            
            return np.clip(normal_data, 0.0, 1.0)
        except Exception as e:
            print(f"Warning: Failed to load normal map {full_path}: {e}")
            return None
                
    def _save_processed_normal_map(self, normal_data, mat_name):
        """
        保存处理后的法线贴图
        返回保存的文件名
        """
        clean_name = re.sub(r'[^\w]', '_', mat_name)[:30]
        output_filename = f"{clean_name}_normal.png"
        
        if self.output_dir:
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            output_path = output_filename
        
        img_uint8 = (normal_data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'RGB')
        img.save(output_path, 'PNG')
        print(f"  Saved processed normal map: {output_path}")
        
        return output_filename
    
    def _load_color_texture_exr(self, exr_path, base_dir):
        """加载EXR颜色贴图"""
        full_path = os.path.join(base_dir, exr_path)
        if not os.path.exists(full_path):
            return None
        
        try:
            data = self._read_exr_file(full_path)
            if data is None:
                return None
            return np.clip(data, 0.0, 1.0)
        except Exception as e:
            print(f"  Warning: Failed to load EXR color texture {exr_path}: {e}")
            return None

    def _save_color_texture_png(self, color_data, mat_name):
        """保存颜色贴图为PNG"""
        clean_name = re.sub(r'[^\w]', '_', mat_name)[:30]
        output_filename = f"{clean_name}_color.png"
        
        if self.output_dir:
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            output_path = output_filename
        
        img_uint8 = (color_data * 255).clip(0, 255).astype(np.uint8)
        
        if len(color_data.shape) == 3:
            img = Image.fromarray(img_uint8, 'RGB')
        else:
            img = Image.fromarray(img_uint8, 'L')
        
        img.save(output_path, 'PNG')
        print(f"  Saved converted color texture: {output_path}")
        
        return output_filename

    def _save_converted_exr(self, channel_name, exr_path, data, mat_name, target_size):
        """保存转换后的EXR为单独的PNG文件"""
        clean_name = re.sub(r'[^\w]', '_', mat_name)[:30]
        output_filename = f"{clean_name}_{channel_name}.png"
        
        if self.output_dir:
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            output_path = output_filename
        
        if data.shape != target_size:
            img_pil = Image.fromarray((data * 255).clip(0, 255).astype(np.uint8))
            img_pil = img_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
            data = np.array(img_pil, dtype=np.float32) / 255.0
        
        img_uint8 = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'L') 
        img.save(output_path, 'PNG')
        print(f"  Saved converted EXR ({channel_name}) to: {output_path}")
    
    def _has_light_keywords(self, mat_name):
        """检查材质名称是否包含光源关键词"""
        mat_name_lower = mat_name.lower()
        light_keywords = [
            'light', 'lamp', 'bulb', 'emissive', 'emit', 'glow', 
            'neon', 'led', 'spot', 'point', 'area', 'sun'
        ]
        return any(keyword in mat_name_lower for keyword in light_keywords)
    
    def is_emissive(self, mat_name):
        """判断材质是否发光"""
        if mat_name not in self.materials:
            return False
        
        mat = self.materials[mat_name]
        
        if self._has_light_keywords(mat_name):
            return True
        
        if 'map_ke' in mat:
            return True
        
        if 'ke' in mat:
            ke_sum = sum(mat['ke'])
            if ke_sum > 0.01:
                return True
        
        if 'illum' in mat:
            illum_mode = mat['illum']
            if illum_mode == 0:
                return True
        
        ka = mat.get('ka', [0.0, 0.0, 0.0])
        kd = mat.get('kd', [0.0, 0.0, 0.0])
        ks = mat.get('ks', [0.0, 0.0, 0.0])
        
        ka_sum = sum(ka)
        kd_sum = sum(kd)
        ks_sum = sum(ks)
        
        if ka_sum > 2.8 and kd_sum > 2.8 and ks_sum > 2.8:
            ka_max = max(ka)
            kd_max = max(kd)
            ks_max = max(ks)
            
            if ka_max > 0.9 and kd_max > 0.9 and ks_max > 0.9:
                return True
        
        d_value = mat.get('d', 1.0)
        tr_value = mat.get('tr', 0.0)
        if d_value < 0.5 or tr_value > 0.5:
            return False
        
        ni_value = mat.get('ni', 1.0)
        if ni_value > 1.1:
            return False
        
        return False
    
    def get_emission_color(self, mat_name):
        """获取自发光颜色"""
        if mat_name not in self.materials:
            return [0.0, 0.0, 0.0]
        
        mat = self.materials[mat_name]
        
        if 'ke' in mat:
            ke = mat['ke']
            ke_sum = sum(ke)
            if ke_sum > 0.001:
                return ke
        
        if 'kd' in mat:
            return mat['kd']
        
        if 'ka' in mat:
            return mat['ka']
        
        return [0.0, 0.0, 0.0]
    
    def get_emission_intensity(self, mat_name):
        """计算发光强度"""
        if mat_name not in self.materials:
            return 0.0
        
        mat = self.materials[mat_name]
        
        if 'ke' in mat:
            ke_sum = sum(mat['ke'])
            if ke_sum > 0.001:
                return ke_sum
        
        if 'kd' in mat:
            kd_sum = sum(mat['kd'])
            if kd_sum > 2.5:
                return kd_sum
        
        return 0.0

    def _merge_textures_to_jpg(self, roughness_tex, specular_tex, metallic_tex, transparency_tex, 
                            base_dir, mat_name, default_roughness, default_specular, 
                            default_metallic, default_transparency):
        """将多张纹理合并成一张4通道PNG并导出"""
        
        # 通道顺序：r=roughness, g=specular, b=metallic, a=opacity 
        texture_keys = [
            ('roughness', roughness_tex, default_roughness),
            ('specular', specular_tex, default_specular),
            ('metallic', metallic_tex, default_metallic),
            ('opacity', transparency_tex, 1.0 - default_transparency) 
        ]
        
        textures = []
        sizes = []
        converted_files = [] 
        
        for channel_name, tex_path, default_val in texture_keys:
            if tex_path and self._check_texture_exists(tex_path, base_dir):
                arr = self._load_texture(tex_path, base_dir)
                if arr is not None:
                    if channel_name == 'opacity':
                        arr = 1.0 - arr
                    textures.append(arr)
                    sizes.append(arr.shape)
                    print(f"  Loaded {channel_name} from {tex_path}: shape={arr.shape}")
                    
                    if tex_path.lower().endswith(('.exr', '.dds')):
                        converted_files.append((channel_name, tex_path, arr))
                    continue
            if tex_path and not self._check_texture_exists(tex_path, base_dir):
                print(f"  Using default {channel_name}={default_val} (texture not found: {tex_path})")
            else:
                print(f"  Using default {channel_name}={default_val}")
            textures.append(None)
            sizes.append(None)
        
        target_size = None
        for size in sizes:
            if size is not None:
                target_size = size
                break
        
        if target_size is None:
            target_size = (512, 512)
            print(f"  No textures found, creating {target_size} default map")
        else:
            print(f"  Target size: {target_size}")
        
        channels = []
        for i, (tex_name, _, _) in enumerate(texture_keys):
            if textures[i] is not None:
                tex = textures[i]
                if tex.shape != target_size:
                    print(f"  Resizing {tex_name} from {tex.shape} to {target_size}")
                    img_pil = Image.fromarray((tex * 255).clip(0, 255).astype(np.uint8))
                    img_pil = img_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
                    tex = np.array(img_pil, dtype=np.float32) / 255.0
                channels.append(tex)
            else:
                _, _, default_val = texture_keys[i]
                channels.append(np.full(target_size, default_val, dtype=np.float32))
        
        # 合并通道 (r=roughness, g=specular, b=metallic, a=opacity)
        combined = np.stack(channels, axis=-1)
        
        clean_name = re.sub(r'[^\w]', '_', mat_name)[:30]
        output_filename = f"{clean_name}_attribute.png"
        if self.output_dir:
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            output_path = output_filename
        
        img_uint8 = (combined * 255).clip(0, 255).astype(np.uint8)
        
        if combined.shape[-1] == 4:
            img = Image.fromarray(img_uint8, 'RGBA')
        else:
            img = Image.fromarray(img_uint8, 'RGB')
        
        img.save(output_path, 'PNG')
        print(f"  Saved attribute map to: {output_path}")
        
        for channel_name, original_path, arr in converted_files:
            if original_path.lower().endswith('.exr'):
                self._save_converted_exr(channel_name, original_path, arr, mat_name, target_size)
            elif original_path.lower().endswith('.dds'):
                self._save_converted_dds_to_png(channel_name, original_path, arr, mat_name, target_size)
        
        return output_filename

    def _get_default_params(self):
        """返回默认材质参数"""
        return {
            'base_color': [0.5, 0.5, 0.5],
            'roughness': 0.5,
            'metallic': 0.0,
            'specular': 0.5,
            'emission': [0.0, 0.0, 0.0],
            'ior': 1.01,
            'transparency': 0.0,
            'subsurface': 0.0,
            'specular_tint': 0.0,
            'anisotropic': 0.0,
            'sheen': 0.0,
            'sheen_tint': 0.0,
            'clearcoat': 0.0,
            'clearcoat_roughness': 0.0,
            'A': 0.0, 'B': 0.0, 'C': 0.0,
            'medium_id': 0,
            'light_index': -1,
            'attribute_texture_file': '',
            'color_texture_file': '',
            'normal_texture_file': '',
            'bump_multiplier': 1.0
        }
    
    def get_material_params(self, mat_name, is_light=False):
        """获取材质参数并合并纹理"""
        if mat_name not in self.materials:
            return self._get_default_params()
        
        mat = self.materials[mat_name]
        params = self._get_default_params()
        base_dir = getattr(self, '_base_dir', '')
        
        roughness_tex = (mat.get('map_roughness') or 
                        mat.get('map_rough') or 
                        mat.get('map_pr') or 
                        mat.get('map_Pr') or
                        mat.get('map_ns') or
                        mat.get('map_Roughness') or '')
        
        metallic_tex = (mat.get('map_metallic') or 
                    mat.get('map_refl') or 
                    mat.get('map_Metallic') or 
                    mat.get('map_metal') or
                    mat.get('map_m') or '')
        
        transparency_tex = (mat.get('map_d') or 
                        mat.get('map_tr') or 
                        mat.get('map_opacity') or
                        mat.get('map_alpha') or
                        mat.get('map_map_d') or '')
        
        specular_tex = (mat.get('map_Ks') or 
                    mat.get('map_specular') or 
                    mat.get('map_spec') or
                    mat.get('map_ks') or '')
        
        color_tex = mat.get('map_kd', '')
        if color_tex and self._check_texture_exists(color_tex, base_dir):
            if color_tex.lower().endswith(('.exr', '.dds')):
                if color_tex.lower().endswith('.exr'):
                    color_data = self._load_color_texture_exr(color_tex, base_dir)
                else:
                    color_data = self._read_dds_file(os.path.join(base_dir, color_tex))
                
                if color_data is not None:
                    params['color_texture_file'] = self._save_color_texture_png(color_data, mat_name)
                else:
                    params['color_texture_file'] = ''
                    print(f"  Warning: Failed to convert color texture {color_tex}, using default color")
            else:
                params['color_texture_file'] = self._copy_texture_to_output(color_tex, base_dir)
        else:
            if color_tex:
                print(f"  Warning: Color texture {color_tex} not found, using default color")
            params['color_texture_file'] = ''
        
        normal_tex = mat.get('map_bump', '')
        bump_multiplier = mat.get('bump_multiplier', 1.0)
        
        if normal_tex and self._check_texture_exists(normal_tex, base_dir):
            if normal_tex.lower().endswith(('.exr', '.dds')) or bump_multiplier != 1.0:
                normal_data = self._load_and_process_normal_map(normal_tex, base_dir, bump_multiplier)
                if normal_data is not None:
                    params['normal_texture_file'] = self._save_processed_normal_map(normal_data, mat_name)
                    params['bump_multiplier'] = 1.0
                else:
                    params['normal_texture_file'] = ''
                    params['bump_multiplier'] = bump_multiplier
                    print(f"  Warning: Failed to convert normal map {normal_tex}, using default normal")
            else:
                params['normal_texture_file'] = self._copy_texture_to_output(normal_tex, base_dir)
                params['bump_multiplier'] = bump_multiplier
        else:
            if normal_tex:
                print(f"  Warning: Normal map {normal_tex} not found, using default normal")
            params['normal_texture_file'] = ''
            params['bump_multiplier'] = 1.0
        
        has_attrib_textures = any([roughness_tex, specular_tex, metallic_tex, transparency_tex])
        
        valid_attrib_textures = []
        for tex in [roughness_tex, specular_tex, metallic_tex, transparency_tex]:
            if tex and self._check_texture_exists(tex, base_dir):
                valid_attrib_textures.append(tex)
        
        has_attrib_textures = any(valid_attrib_textures)
        
        if has_attrib_textures and self.output_dir:
            base_dir = getattr(self, '_base_dir', '')
            
            default_roughness = 0.5
            default_specular = 0.5
            default_metallic = 0.0
            default_transparency = 0.0
            
            if 'ns' in mat:
                default_roughness = pow(2.0 / (mat['ns'] + 2), 0.5)
            
            if 'kd' in mat:
                kd_sum = sum(mat['kd'])
                if kd_sum > 2.7:
                    default_metallic = 0.9
            
            if 'd' in mat:
                default_transparency = 1.0 - mat['d']
            elif 'tr' in mat:
                default_transparency = mat['tr']
            
            merged_filename = self._merge_textures_to_jpg(
                roughness_tex, specular_tex, metallic_tex, transparency_tex,
                base_dir, mat_name, default_roughness, default_specular,
                default_metallic, default_transparency
            )
            
            params['attribute_texture_file'] = merged_filename
            print(f"Merged attribute textures for {mat_name} -> {merged_filename}")
        else:
            pass
        
        if 'kd' in mat:
            params['base_color'] = mat['kd']
        
        if 'ns' in mat:
            params['roughness'] = pow(2.0 / (mat['ns'] + 2), 0.5)
        
        if 'kd' in mat:
            kd_sum = sum(mat['kd'])
            if kd_sum > 2.7:
                params['metallic'] = 0.9
        
        if 'd' in mat:
            params['transparency'] = 1.0 - mat['d']
        elif 'tr' in mat:
            params['transparency'] = mat['tr']
        
        if 'ke' in mat:
            params['emission'] = mat['ke']
        
        if is_light:
            params['light_index'] = len([m for m in self.materials if self.is_emissive(m)])
        
        known_keys = {'newmtl', 'ka', 'kd', 'ks', 'ke', 'tf', 'd', 'tr', 'ns', 'ni', 'illum',
                    'map_kd', 'map_bump', 'map_roughness', 'map_pr', 'map_rough', 
                    'map_ns', 'map_metallic', 'map_d', 'map_tr', 'map_ke',
                    'bump_multiplier', 'map_Roughness', 'map_Metallic', 'map_Specular',
                    'map_opacity', 'map_alpha', 'map_Opacity', 'map_Ks', 'map_spec', 'map_metal', 'map_m',
                    'map_refl'}
        for key in mat.keys():
            if key.startswith('map_') and key not in known_keys:
                print(f"Warning: Unrecognized texture type in {mat_name}: {key} = {mat[key]}")
        
        return params
    
    def set_base_dir(self, base_dir):
        """设置纹理的基础目录"""
        self._base_dir = base_dir

class OBJSplitter:
    """拆分OBJ文件"""
    def __init__(self, obj_path, output_dir="output"):
        self.obj_path = obj_path
        self.output_dir = output_dir
        self.mtl_parser = None
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.content = f.read()
        
        mtl_match = re.search(r'mtllib\s+(.+)', self.content)
        if mtl_match:
            mtl_filename = mtl_match.group(1).strip()
            obj_dir = os.path.dirname(obj_path)
            mtl_path = os.path.join(obj_dir, mtl_filename)
            self.mtl_parser = MTLParser(mtl_path, output_dir)
            self.mtl_parser.set_base_dir(obj_dir)
    
    def parse_obj(self):
        """解析OBJ文件，按材质分组"""
        lines = self.content.split('\n')
        
        vertices = []
        texcoords = []
        normals = []
        faces_by_material = defaultdict(list)
        
        current_material = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            command = parts[0].lower()
            
            if command == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif command == 'vt':
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif command == 'vn':
                normals.append([float(x) for x in parts[1:4]])
            elif command == 'usemtl':
                current_material = parts[1]
            elif command == 'f':
                if current_material:
                    face_data = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        v_idx = int(indices[0]) if indices[0] else 0
                        vt_idx = int(indices[1]) if len(indices) > 1 and indices[1] else 0
                        vn_idx = int(indices[2]) if len(indices) > 2 and indices[2] else 0
                        face_data.append((v_idx, vt_idx, vn_idx))
                    
                    faces_by_material[current_material].append(face_data)
        
        return vertices, texcoords, normals, faces_by_material
    
    def write_split_obj(self, mat_name, faces, vertices, texcoords, normals, is_light=False):
        """写入拆分后的OBJ文件"""
        used_v_indices = set()
        used_vt_indices = set()
        used_vn_indices = set()
        
        for face in faces:
            for v_idx, vt_idx, vn_idx in face:
                if v_idx > 0:
                    used_v_indices.add(v_idx)
                if vt_idx > 0:
                    used_vt_indices.add(vt_idx)
                if vn_idx > 0:
                    used_vn_indices.add(vn_idx)
        
        v_map = {old_idx: new_idx+1 for new_idx, old_idx in enumerate(sorted(used_v_indices))}
        vt_map = {old_idx: new_idx+1 for new_idx, old_idx in enumerate(sorted(used_vt_indices))}
        vn_map = {old_idx: new_idx+1 for new_idx, old_idx in enumerate(sorted(used_vn_indices))}
        
        base_name = os.path.splitext(os.path.basename(self.obj_path))[0]
        mat_name_clean = re.sub(r'[^\w]', '_', mat_name)
        if is_light:
            output_filename = f"{base_name}_{mat_name_clean}_light.obj"
        else:
            output_filename = f"{base_name}_{mat_name_clean}.obj"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if self.mtl_parser:
                f.write(f"mtllib {os.path.basename(self.obj_path).replace('.obj', '.mtl')}\n")
                f.write(f"usemtl {mat_name}\n")
                f.write("\n")
            
            for v_idx in sorted(used_v_indices):
                v = vertices[v_idx - 1]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            for vt_idx in sorted(used_vt_indices):
                vt = texcoords[vt_idx - 1]
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
            
            for vn_idx in sorted(used_vn_indices):
                vn = normals[vn_idx - 1]
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            
            for face in faces:
                face_str_parts = []
                for v_idx, vt_idx, vn_idx in face:
                    if vt_idx > 0 and vn_idx > 0:
                        face_str = f"{v_map[v_idx]}/{vt_map[vt_idx]}/{vn_map[vn_idx]}"
                    elif vt_idx > 0:
                        face_str = f"{v_map[v_idx]}/{vt_map[vt_idx]}"
                    elif vn_idx > 0:
                        face_str = f"{v_map[v_idx]}//{vn_map[vn_idx]}"
                    else:
                        face_str = str(v_map[v_idx])
                    face_str_parts.append(face_str)
                f.write(f"f {' '.join(face_str_parts)}\n")
        
        return output_path
    
    def split(self):
        """拆分OBJ文件"""
        vertices, texcoords, normals, faces_by_material = self.parse_obj()
        
        results = []
        light_results = []
        
        for mat_name, faces in faces_by_material.items():
            if self.mtl_parser:
                is_light = self.mtl_parser.is_emissive(mat_name)
                
                if is_light:
                    intensity = self.mtl_parser.get_emission_intensity(mat_name)
                    emission_color = self.mtl_parser.get_emission_color(mat_name)
                    
                    print(f"\n[光源检测] 材质: {mat_name}")
                    print(f"  发光强度: {intensity:.4f}")
                    print(f"  发光颜色: R={emission_color[0]:.3f} G={emission_color[1]:.3f} B={emission_color[2]:.3f}")
                    
                    output_path = self.write_split_obj(
                        mat_name, faces, vertices, texcoords, normals, is_light=True
                    )
                    light_results.append({
                        'material': mat_name,
                        'intensity': intensity,
                        'emission': emission_color,
                        'file': output_path
                    })
                else:
                    output_path = self.write_split_obj(
                        mat_name, faces, vertices, texcoords, normals, is_light=False
                    )
                    results.append({
                        'material': mat_name,
                        'file': output_path
                    })
            else:
                output_path = self.write_split_obj(
                    mat_name, faces, vertices, texcoords, normals, is_light=False
                )
                results.append({
                    'material': mat_name,
                    'file': output_path
                })
        
        return results, light_results


def split_obj_file(obj_path, output_dir="output"):
    """
    拆分OBJ文件的便捷函数
    """
    print(f"处理OBJ文件: {obj_path}")
    splitter = OBJSplitter(obj_path, output_dir)
    meshes, lights = splitter.split()
    
    print(f"\n拆分完成!")
    print(f"普通物体: {len(meshes)} 个")
    print(f"光源物体: {len(lights)} 个")
    
    return meshes, lights


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python obj_splitter.py <obj_file> [output_dir]")
        sys.exit(1)
    
    obj_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    split_obj_file(obj_file, output_dir)
