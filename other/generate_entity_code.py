import os
import re
from obj_splitter import MTLParser

def generate_cpp_code_with_params(meshes, lights, scale=1.0, base_position=(0.0, 0.0, 0.0)):
    """
    Generate C++ code using pre-computed material params
    
    Args:
        meshes: List of regular meshes (with 'params' key)
        lights: List of lights (with 'params' key)
        scale: Scale factor
        base_position: Base position (x, y, z)
    
    Returns:
        C++ code string
    """
    cpp_code = []
    
    cpp_code.append("// ========== Auto-generated entity code ==========")
    cpp_code.append("// Meshes: {}, Lights: {}".format(len(meshes), len(lights)))
    cpp_code.append("")
    
    cpp_code.append("// ========== Regular objects ==========")
    for i, mesh in enumerate(meshes):
        mat_name = mesh['material']
        filename = os.path.basename(mesh['file'])
        params = mesh.get('params', {})
        
        mat_name_clean = re.sub(r'[^\w]', '_', mat_name)[:20]
        if mat_name_clean[0].isdigit():
            mat_name_clean = 'm_' + mat_name_clean
        
        cpp_code.append("{")
        
        texture_lines = _get_texture_loading_lines(params)
        for line in texture_lines:
            cpp_code.append("    {}".format(line))
        
        cpp_code.append("    Material {}_material(".format(mat_name_clean))
        _write_material_params_lines(cpp_code, params)
        cpp_code.append("    );")
        
        cpp_code.append("    auto {} = std::make_shared<Entity>( ".format(mat_name_clean))
        cpp_code.append('        "meshes/{}",'.format(filename))
        cpp_code.append("        {}_material,".format(mat_name_clean))
        cpp_code.append("        glm::scale(")
        cpp_code.append("            glm::translate(glm::mat4(1.0f), glm::vec3({:.2f}, {:.2f}, {:.2f})),".format(
            base_position[0], base_position[1], base_position[2]))
        cpp_code.append("            glm::vec3({:.2f})".format(scale))
        cpp_code.append("        )")
        cpp_code.append("    );")
        cpp_code.append("    scene_->AddEntity({});".format(mat_name_clean))
        cpp_code.append("}")
        cpp_code.append("")
    
    if lights:
        cpp_code.append("// ========== Lights (emissive entities) ==========")
        for i, light in enumerate(lights):
            mat_name = light['material']
            filename = os.path.basename(light['file'])
            intensity = light['intensity']
            emission = light['emission']
            params = light.get('params', {})
            
            if sum(emission) > 0.01:
                params['emission'] = emission
            else:
                params['emission'] = params.get('base_color', [0.0, 0.0, 0.0])
            
            scaled_emission = [
                params['emission'][0] * max(intensity, 1.0),
                params['emission'][1] * max(intensity, 1.0),
                params['emission'][2] * max(intensity, 1.0)
            ]
            
            mat_name_clean = re.sub(r'[^\w]', '_', mat_name)[:20]
            if mat_name_clean[0].isdigit():
                mat_name_clean = 'm_' + mat_name_clean
            entity_name = 'light_' + mat_name_clean
            
            cpp_code.append("{{")
            cpp_code.append("    // Light: {} (intensity: {:.4f})".format(mat_name, intensity))
            
            texture_lines = _get_texture_loading_lines(params)
            for line in texture_lines:
                cpp_code.append("    {}".format(line))
            
            cpp_code.append("    Material {}_material(".format(entity_name))
            _write_material_params_lines(cpp_code, params, emission_override=scaled_emission)
            cpp_code.append("    );")
            
            cpp_code.append("    auto {} = std::make_shared<Entity>( ".format(entity_name))
            cpp_code.append('        "meshes/{}",'.format(filename))
            cpp_code.append("        {}_material,".format(entity_name))
            cpp_code.append("        glm::scale(")
            cpp_code.append("            glm::translate(glm::mat4(1.0f), glm::vec3({:.2f}, {:.2f}, {:.2f})),".format(
                base_position[0], base_position[1], base_position[2]))
            cpp_code.append("            glm::vec3({:.2f})".format(scale))
            cpp_code.append("        )")
            cpp_code.append("    );")
            cpp_code.append("    scene_->AddEntity({});".format(entity_name))
            cpp_code.append("}}")
            cpp_code.append("")
    
    return "\n".join(cpp_code)


def _get_texture_loading_lines(params):
    """
    Generate texture loading code lines
    Returns list of strings
    """
    lines = []
    
    if params['color_texture_file']:
        lines.append('int colorTexId = texture_manager_->LoadTexture("textures/{}");'.format(params['color_texture_file']))
    
    if params['normal_texture_file']:
        lines.append('int normalTexId = texture_manager_->LoadTexture("textures/{}");'.format(params['normal_texture_file']))
    
    if params['attribute_texture_file']:
        lines.append('int attribTexId = texture_manager_->LoadTexture("textures/{}");'.format(params['attribute_texture_file']))
    
    return lines


def _write_material_params_lines(cpp_code, params, emission_override=None):
    """Write material parameters to C++ code"""
    
    emission_val = emission_override if emission_override else params['emission']
    base_color = params['base_color'] if emission_override is None else [0.0, 0.0, 0.0]
    
    tex_color = 'colorTexId' if params['color_texture_file'] else -1
    tex_normal = 'normalTexId' if params['normal_texture_file'] else -1
    tex_attrib = 'attribTexId' if params['attribute_texture_file'] else -1
    
    cpp_code.append("        glm::vec3({:.3f}, {:.3f}, {:.3f}),  // base_color".format(
        base_color[0], base_color[1], base_color[2]))
    cpp_code.append("        {:.3f},  // roughness".format(params['roughness']))
    cpp_code.append("        {:.2f},  // metallic".format(params['metallic']))
    cpp_code.append("")
    cpp_code.append("        // Lighting parameters")
    cpp_code.append("        0xFFFFFFFF,  // light_index")
    cpp_code.append("        glm::vec3({:.3f}, {:.3f}, {:.3f}),  // emission".format(
        emission_val[0], emission_val[1], emission_val[2]))
    cpp_code.append("")
    cpp_code.append("        // Optical parameters")
    cpp_code.append("        {:.3f},  // ior".format(params['ior']))
    cpp_code.append("        {:.2f},  // transparency".format(params['transparency']))
    cpp_code.append("")
    cpp_code.append("        // Texture indices")
    cpp_code.append("        {},  // texture_id".format(tex_color))
    cpp_code.append("        {},  // normal_tex_id".format(tex_normal))
    cpp_code.append("        {},  // attribute_tex_id (r=roughness, g=specular, b=metallic, a=opacity)".format(tex_attrib))
    cpp_code.append("")
    cpp_code.append("        // Subsurface scattering")
    cpp_code.append("        {:.2f},  // subsurface".format(params['subsurface']))
    cpp_code.append("")
    cpp_code.append("        // Specular reflection parameters")
    cpp_code.append("        {:.2f},  // specular".format(params['specular']))
    cpp_code.append("        {:.2f},  // specular_tint".format(params['specular_tint']))
    cpp_code.append("")
    cpp_code.append("        // Anisotropy")
    cpp_code.append("        {:.2f},  // anisotropic".format(params['anisotropic']))
    cpp_code.append("")
    cpp_code.append("        // Sheen layer")
    cpp_code.append("        {:.2f},  // sheen".format(params['sheen']))
    cpp_code.append("        {:.2f},  // sheen_tint".format(params['sheen_tint']))
    cpp_code.append("")
    cpp_code.append("        // Clearcoat layer")
    cpp_code.append("        {:.2f},  // clearcoat".format(params['clearcoat']))
    cpp_code.append("        {:.2f},  // clearcoat_roughness".format(params['clearcoat_roughness']))
    cpp_code.append("")
    cpp_code.append("        // Dispersion coefficients (Cauchy equation)")
    cpp_code.append("        {:.2f}, {:.2f}, {:.2f},  // A, B, C (dispersion)".format(
        params['A'], params['B'], params['C']))
    cpp_code.append("        {}  // medium_id".format(params['medium_id']))


def save_cpp_code(cpp_code, output_file="generated_entities.cpp"):
    """Save C++ code to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cpp_code)
    print("\nC++ code saved to: {}".format(output_file))


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_entity_code.py <obj_file> [output_dir] [scale] [pos_x] [pos_y] [pos_z]")
        print("Example: python generate_entity_code.py model.obj output 10.0 0.0 0.0 0.0")
        sys.exit(1)
    
    obj_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    pos_x = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    pos_y = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    pos_z = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    
    mtl_path = None
    with open(obj_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        mtl_match = re.search(r'mtllib\s+(.+)', content)
        if mtl_match:
            mtl_filename = mtl_match.group(1).strip()
            obj_dir = os.path.dirname(obj_file)
            mtl_path = os.path.join(obj_dir, mtl_filename)
    
    from obj_splitter import split_obj_file
    meshes, lights = split_obj_file(obj_file, output_dir)
    
    if mtl_path and os.path.exists(mtl_path):
        from obj_splitter import MTLParser
        mtl_parser = MTLParser(mtl_path, output_dir)
        mtl_parser.set_base_dir(os.path.dirname(obj_file))
        
        all_materials = set()
        for mesh in meshes:
            all_materials.add(mesh['material'])
        for light in lights:
            all_materials.add(light['material'])
        
        for mat_name in all_materials:
            params = mtl_parser.get_material_params(mat_name, is_light=False)
            for light in lights:
                if light['material'] == mat_name:
                    mtl_parser.get_material_params(mat_name, is_light=True)
                    break
        
        for mesh in meshes:
            params = mtl_parser.get_material_params(mesh['material'], is_light=False)
            mesh['params'] = params
        
        for light in lights:
            params = mtl_parser.get_material_params(light['material'], is_light=True)
            light['params'] = params
        
        cpp_code = generate_cpp_code_with_params(meshes, lights, scale, (pos_x, pos_y, pos_z))
        print("Using MTL file: {}".format(mtl_path))
    else:
        print("Warning: MTL file not found, using default material parameters")
        cpp_code = generate_cpp_code_with_params(meshes, lights, scale, (pos_x, pos_y, pos_z))
    
    cpp_output_file = os.path.join(output_dir, "generated_entities.cpp")
    save_cpp_code(cpp_code, cpp_output_file)
    
    if lights:
        print("\n" + "="*60)
        print("Light Summary:")
        print("="*60)
        for i, light in enumerate(lights):
            mat_name = light['material']
            emission = light['emission']
            intensity = light['intensity']
            
            print("{}. Material: {}".format(i+1, mat_name))
            print("   Intensity: {:.4f}".format(intensity))
            print("   Color: R={:.3f} G={:.3f} B={:.3f}".format(emission[0], emission[1], emission[2]))
            print("")

if __name__ == "__main__":
    main()
