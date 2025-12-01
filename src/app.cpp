#include "app.h"
#include "Material.h"
#include "Entity.h"

#include "glm/gtc/matrix_transform.hpp"
#include "imgui.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace {
#include "built_in_shaders.inl"
}

Application::Application(grassland::graphics::BackendAPI api) 
    : frame_count_(0)
    , samples_per_pixel_(1) {

    grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{}, &core_);
    core_->InitializeLogicalDeviceAutoSelect(true);

    grassland::LogInfo("Device Name: {}", core_->DeviceName());
    grassland::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
    core_.reset();
}

// Event handler for keyboard input
void Application::ProcessInput() {
    GLFWwindow* glfw_window = window_->GLFWWindow();
    
    if (glfwGetWindowAttrib(glfw_window, GLFW_FOCUSED) == GLFW_FALSE) {
        return;
    }

    // Tab key to toggle UI visibility (only in inspection mode)
    if (!camera_enabled_) {
        ui_hidden_ = (glfwGetKey(glfw_window, GLFW_KEY_TAB) == GLFW_PRESS);
    }
    
    // Ctrl+S to save accumulated output (only in inspection mode)
    static bool ctrl_s_was_pressed = false;
    bool ctrl_pressed = (glfwGetKey(glfw_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || 
                        glfwGetKey(glfw_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    bool s_pressed = (glfwGetKey(glfw_window, GLFW_KEY_S) == GLFW_PRESS);
    bool ctrl_s_pressed = ctrl_pressed && s_pressed;
    
    if (ctrl_s_pressed && !ctrl_s_was_pressed && !camera_enabled_) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_s(&tm, &time_t);
        
        std::ostringstream filename;
        filename << "screenshot_" 
                 << std::put_time(&tm, "%Y%m%d_%H%M%S")
                 << ".png";
        
        SaveAccumulatedOutput(filename.str());
    }
    ctrl_s_was_pressed = ctrl_s_pressed;
    
    // Only process camera movement if camera is enabled
    if (!camera_enabled_) {
        return;
    }

    // Poll key states directly
    if (glfwGetKey(glfw_window, GLFW_KEY_W) == GLFW_PRESS) {
        camera_pos_ += camera_speed_ * camera_front_;
    }
    if (glfwGetKey(glfw_window, GLFW_KEY_S) == GLFW_PRESS) {
        camera_pos_ -= camera_speed_ * camera_front_;
    }
    if (glfwGetKey(glfw_window, GLFW_KEY_A) == GLFW_PRESS) {
        camera_pos_ -= glm::normalize(glm::cross(camera_front_, camera_up_)) * camera_speed_;
    }
    if (glfwGetKey(glfw_window, GLFW_KEY_D) == GLFW_PRESS) {
        camera_pos_ += glm::normalize(glm::cross(camera_front_, camera_up_)) * camera_speed_;
    }
    if (glfwGetKey(glfw_window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera_pos_ += camera_speed_ * camera_up_;
    }
    if (glfwGetKey(glfw_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
        glfwGetKey(glfw_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        camera_pos_ -= camera_speed_ * camera_up_;
    }
}

void Application::OnMouseMove(double xpos, double ypos) {
    mouse_x_ = xpos;
    mouse_y_ = ypos;

    if (!camera_enabled_) {
        return;
    }

    if (first_mouse_) {
        last_x_ = (float)xpos;
        last_y_ = (float)ypos;
        first_mouse_ = false;
        return;
    }

    float xoffset = (float)xpos - last_x_;
    float yoffset = last_y_ - (float)ypos;
    last_x_ = (float)xpos;
    last_y_ = (float)ypos;

    xoffset *= mouse_sensitivity_;
    yoffset *= mouse_sensitivity_;

    yaw_ += xoffset;
    pitch_ += yoffset;

    if (pitch_ > 89.0f)
        pitch_ = 89.0f;
    if (pitch_ < -89.0f)
        pitch_ = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    camera_front_ = glm::normalize(front);
}

void Application::OnMouseButton(int button, int action, int mods, double xpos, double ypos) {
    const int BUTTON_LEFT = 0;
    const int BUTTON_RIGHT = 1;
    const int ACTION_PRESS = 1;

    if (button == BUTTON_LEFT && action == ACTION_PRESS && !camera_enabled_) {
        if (hovered_entity_id_ >= 0) {
            selected_entity_id_ = hovered_entity_id_;
            grassland::LogInfo("Selected Entity #{}", selected_entity_id_);
        } else {
            selected_entity_id_ = -1;
            grassland::LogInfo("Deselected entity");
        }
    }

    if (button == BUTTON_RIGHT && action == ACTION_PRESS) {
        camera_enabled_ = !camera_enabled_;
        
        GLFWwindow* glfw_window = window_->GLFWWindow();
        
        if (camera_enabled_) {
            glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            first_mouse_ = true;
            grassland::LogInfo("Camera mode enabled - use WASD/Space/Shift to move, mouse to look");
        } else {
            glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            grassland::LogInfo("Camera mode disabled - cursor visible, showing info overlay");
        }
    }
}

void Application::OnInit() {
    alive_ = true;
    core_->CreateWindowObject(2560, 1440,
        ((core_->API() == grassland::graphics::BACKEND_API_VULKAN) ? "[Vulkan]" : "[D3D12]") +
        std::string(" Ray Tracing Scene Demo"),
        &window_);

    window_->InitImGui();

    window_->MouseMoveEvent().RegisterCallback(
        [this](double xpos, double ypos) {
            this->OnMouseMove(xpos, ypos);
        }
    );
    window_->MouseButtonEvent().RegisterCallback(
        [this](int button, int action, int mods, double xpos, double ypos) {
            this->OnMouseButton(button, action, mods, xpos, ypos);
        }
    );

    camera_enabled_ = false;
    last_camera_enabled_ = false;
    ui_hidden_ = false;
    hovered_entity_id_ = -1;
    hovered_pixel_color_ = glm::vec4(0.0f);
    selected_entity_id_ = -1;
    mouse_x_ = 0.0;
    mouse_y_ = 0.0;

    // 初始化光源管理器
    light_manager_ = std::make_unique<LightManager>();
    light_manager_->Initialize(core_.get());
    
    // 添加默认光源
    light_manager_->CreateDefaultLights();
    
    // 初始化渲染设置
    core_->CreateBuffer(sizeof(RenderSettings), grassland::graphics::BUFFER_TYPE_DYNAMIC, &render_settings_buffer_);
    
    RenderSettings initial_settings{};
    initial_settings.frame_count = 0;
    initial_settings.samples_per_pixel = samples_per_pixel_;
    initial_settings.max_depth = 8;
    initial_settings.enable_accumulation = 1;
    render_settings_buffer_->UploadData(&initial_settings, sizeof(RenderSettings));

    // 创建场景
    scene_ = std::make_unique<Scene>(core_.get());

    // 添加实体
    {
        auto ground = std::make_shared<Entity>(
            "meshes/cube.obj",
            Material(glm::vec3(0.8f, 0.8f, 0.8f), 0.8f, 0.0f),
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.0f, 0.0f)), 
                      glm::vec3(10.0f, 0.1f, 10.0f))
        );
        scene_->AddEntity(ground);
    }

    {
        auto red_sphere = std::make_shared<Entity>(
            "meshes/octahedron.obj",
            Material(glm::vec3(1.0f, 0.2f, 0.2f), 0.0f, 0.0f),
                glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.5f, 0.0f))
        );
        scene_->AddEntity(red_sphere);
    }

    {
        auto green_sphere = std::make_shared<Entity>(
            "meshes/preview_sphere.obj",
            Material(glm::vec3(0.8f, 0.95f, 0.8f), 0.2f, 0.8f),
                glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.0f))
        );
        scene_->AddEntity(green_sphere);
    }

    {
        auto blue_cube = std::make_shared<Entity>(
            "meshes/cube.obj",
            Material(glm::vec3(0.2f, 0.2f, 1.0f), 0.5f, 0.0f),
            glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.5f, 0.0f))
        );
        scene_->AddEntity(blue_cube);
    }

    scene_->BuildAccelerationStructures();

    // 创建film
    film_ = std::make_unique<Film>(core_.get(), window_->GetWidth(), window_->GetHeight());

    core_->CreateBuffer(sizeof(CameraObject), grassland::graphics::BUFFER_TYPE_DYNAMIC, &camera_object_buffer_);
    
    core_->CreateBuffer(sizeof(HoverInfo), grassland::graphics::BUFFER_TYPE_DYNAMIC, &hover_info_buffer_);
    HoverInfo initial_hover{};
    initial_hover.hovered_entity_id = -1;
    hover_info_buffer_->UploadData(&initial_hover, sizeof(HoverInfo));

    // 初始化相机状态
    camera_pos_ = glm::vec3{ 0.0f, 1.0f, 5.0f };
    camera_up_ = glm::vec3{ 0.0f, 1.0f, 0.0f };
    camera_speed_ = 0.01f;

    yaw_ = -90.0f;
    pitch_ = 0.0f;
    last_x_ = (float)window_->GetWidth() / 2.0f;
    last_y_ = (float)window_->GetHeight() / 2.0f;
    mouse_sensitivity_ = 0.1f;
    first_mouse_ = true;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    camera_front_ = glm::normalize(front);

    CameraObject camera_object{};
    camera_object.screen_to_camera = glm::inverse(
        glm::perspective(glm::radians(60.0f), (float)window_->GetWidth() / (float)window_->GetHeight(), 0.1f, 10.0f));
    camera_object.camera_to_world =
        glm::inverse(glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_));
    camera_object_buffer_->UploadData(&camera_object, sizeof(CameraObject));

    core_->CreateImage(window_->GetWidth(), window_->GetHeight(), grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
        &color_image_);
    
    core_->CreateImage(window_->GetWidth(), window_->GetHeight(), grassland::graphics::IMAGE_FORMAT_R32_SINT,
        &entity_id_image_);

    core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "RayGenMain", "lib_6_3", &raygen_shader_);
    core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "MissMain", "lib_6_3", &miss_shader_);
    core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "ClosestHitMain", "lib_6_3", &closest_hit_shader_);
    if (!raygen_shader_ || !miss_shader_ || !closest_hit_shader_) {
        grassland::LogError("Failed to create one or more shaders");
        alive_ = false;
        return;
    }
    grassland::LogInfo("Shader compiled successfully");

    core_->CreateRayTracingProgram(raygen_shader_.get(), miss_shader_.get(), closest_hit_shader_.get(), &program_);
    if (!program_) {
        grassland::LogError("Failed to create ray tracing program");
        alive_ = false;
        return;
    }

    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);  // space0
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);          // space1 - color output
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);          // space2
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);          // space3 - materials
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);          // space4 - hover info
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);          // space5 - entity ID output
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);          // space6 - accumulated color
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);          // space7 - accumulated samples

    // 几何信息
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);          // space8 - geometry descriptors
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);          // space9 - vertex infos 
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);          // space10 - indices    

    // 渲染设置
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);          // space 11 - render setting
    
    // 光源数据
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);          // space 12 - lights buffer
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);          // space 13 - light count
    
    try {
        program_->Finalize();
        grassland::LogInfo("Ray tracing program finalized successfully");
    } catch (const std::exception& e) {
        grassland::LogError("Failed to finalize ray tracing program: {}", e.what());
        alive_ = false;
        return;
    }    
}

void Application::OnClose() {
    program_.reset();
    raygen_shader_.reset();
    miss_shader_.reset();
    closest_hit_shader_.reset();

    scene_.reset();
    film_.reset();
    
    light_manager_.reset();

    color_image_.reset();
    entity_id_image_.reset();
    camera_object_buffer_.reset();
    hover_info_buffer_.reset();
    
    window_.reset();
}

void Application::UpdateHoveredEntity() {
    if (camera_enabled_) {
        hovered_entity_id_ = -1;
        hovered_pixel_color_ = glm::vec4(0.0f);
        return;
    }

    int x = static_cast<int>(mouse_x_);
    int y = static_cast<int>(mouse_y_);
    int width = window_->GetWidth();
    int height = window_->GetHeight();
    
    if (x < 0 || x >= width || y < 0 || y >= height) {
        hovered_entity_id_ = -1;
        hovered_pixel_color_ = glm::vec4(0.0f);
        return;
    }

    grassland::graphics::Offset2D offset{ x, y };
    grassland::graphics::Extent2D extent{ 1, 1 };
    
    int32_t entity_id = -1;
    entity_id_image_->DownloadData(&entity_id, offset, extent);
    hovered_entity_id_ = entity_id;
    
    float accumulated_rgba[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    film_->GetAccumulatedColorImage()->DownloadData(accumulated_rgba, offset, extent);
    
    int sample_count = film_->GetSampleCount();
    if (sample_count > 0) {
        hovered_pixel_color_ = glm::vec4(
            accumulated_rgba[0] / static_cast<float>(sample_count),
            accumulated_rgba[1] / static_cast<float>(sample_count),
            accumulated_rgba[2] / static_cast<float>(sample_count),
            accumulated_rgba[3] / static_cast<float>(sample_count)
        );
    } else {
        hovered_pixel_color_ = glm::vec4(0.0f);
    }
}

void Application::OnUpdate() {
    if (window_->ShouldClose()) {
        window_->CloseWindow();
        alive_ = false;
        return;
    }
    if (alive_) {
        ProcessInput();

        if (!camera_enabled_) {
            frame_count_++;
            if( (frame_count_ & 7) == 0)
                grassland::LogInfo("Processing frame {}",frame_count_);
        } else {
            frame_count_ = 0;
        }
    
        RenderSettings settings{};
        settings.frame_count = frame_count_;
        settings.samples_per_pixel = samples_per_pixel_;
        settings.max_depth = 8;
        settings.enable_accumulation = !camera_enabled_;
        
        render_settings_buffer_->UploadData(&settings, sizeof(RenderSettings));

        if (camera_enabled_ != last_camera_enabled_) {
            if (camera_enabled_) {
                grassland::LogInfo("Camera enabled - accumulation will reset when camera stops");
            } else {
                film_->Reset();
                grassland::LogInfo("Camera disabled - starting accumulation");
            }
            last_camera_enabled_ = camera_enabled_;
        }
        
        UpdateHoveredEntity();
        
        HoverInfo hover_info{};
        hover_info.hovered_entity_id = hovered_entity_id_;
        hover_info_buffer_->UploadData(&hover_info, sizeof(HoverInfo));

        CameraObject camera_object{};
        camera_object.screen_to_camera = glm::inverse(
            glm::perspective(glm::radians(60.0f), (float)window_->GetWidth() / (float)window_->GetHeight(), 0.1f, 10.0f));
        camera_object.camera_to_world =
            glm::inverse(glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_));
        camera_object_buffer_->UploadData(&camera_object, sizeof(CameraObject));
    }
}

void Application::ApplyHoverHighlight(grassland::graphics::Image* image) {
    int width = window_->GetWidth();
    int height = window_->GetHeight();
    size_t pixel_count = width * height;
    
    std::vector<float> image_data(pixel_count * 4);
    image->DownloadData(image_data.data());
    
    std::vector<int32_t> entity_ids(pixel_count);
    entity_id_image_->DownloadData(entity_ids.data());
    
    float highlight_factor = 0.4f;
    for (size_t i = 0; i < pixel_count; i++) {
        if (entity_ids[i] == hovered_entity_id_) {
            image_data[i * 4 + 0] = image_data[i * 4 + 0] * (1.0f - highlight_factor) + 1.0f * highlight_factor;
            image_data[i * 4 + 1] = image_data[i * 4 + 1] * (1.0f - highlight_factor) + 1.0f * highlight_factor;
            image_data[i * 4 + 2] = image_data[i * 4 + 2] * (1.0f - highlight_factor) + 1.0f * highlight_factor;
        }
    }
    
    image->UploadData(image_data.data());
}

void Application::SaveAccumulatedOutput(const std::string& filename) {
    int width = window_->GetWidth();
    int height = window_->GetHeight();
    int sample_count = film_->GetSampleCount();
    
    if (sample_count == 0) {
        grassland::LogWarning("Cannot save screenshot: no samples accumulated yet");
        return;
    }
    
    std::vector<float> accumulated_colors(width * height * 4);
    film_->GetAccumulatedColorImage()->DownloadData(accumulated_colors.data());
    
    std::vector<uint8_t> byte_data(width * height * 4);
    for (size_t i = 0; i < width * height; i++) {
        float r = accumulated_colors[i * 4 + 0] / static_cast<float>(sample_count);
        float g = accumulated_colors[i * 4 + 1] / static_cast<float>(sample_count);
        float b = accumulated_colors[i * 4 + 2] / static_cast<float>(sample_count);
        float a = accumulated_colors[i * 4 + 3] / static_cast<float>(sample_count);
        
        byte_data[i * 4 + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, r)) * 255.0f);
        byte_data[i * 4 + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, g)) * 255.0f);
        byte_data[i * 4 + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, b)) * 255.0f);
        byte_data[i * 4 + 3] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, a)) * 255.0f);
    }
    
    int result = stbi_write_png(filename.c_str(), width, height, 4, byte_data.data(), width * 4);
    
    if (result) {
        std::filesystem::path abs_path = std::filesystem::absolute(filename);
        grassland::LogInfo("Screenshot saved: {} ({}x{}, {} samples)", 
                          abs_path.string(), width, height, sample_count);
    } else {
        grassland::LogError("Failed to save screenshot: {}", filename);
    }
}

void Application::RenderInfoOverlay() {
    if (camera_enabled_ || ui_hidden_) {
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(350.0f, (float)window_->GetHeight()), ImGuiCond_Always);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | 
                                     ImGuiWindowFlags_NoResize | 
                                     ImGuiWindowFlags_NoCollapse;
    
    if (!ImGui::Begin("Scene Information", nullptr, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::SeparatorText("Camera");
    ImGui::Text("Position: (%.2f, %.2f, %.2f)", camera_pos_.x, camera_pos_.y, camera_pos_.z);
    ImGui::Text("Direction: (%.2f, %.2f, %.2f)", camera_front_.x, camera_front_.y, camera_front_.z);
    ImGui::Text("Yaw: %.1f°  Pitch: %.1f°", yaw_, pitch_);
    ImGui::Text("Speed: %.3f", camera_speed_);
    ImGui::Text("Sensitivity: %.2f", mouse_sensitivity_);

    ImGui::Spacing();

    ImGui::SeparatorText("Scene");
    size_t entity_count = scene_->GetEntityCount();
    ImGui::Text("Entities: %zu", entity_count);
    ImGui::Text("Materials: %zu", entity_count);
    
    if (light_manager_) {
        size_t light_count = light_manager_->GetLightCount();
        int enabled_lights = light_manager_->GetEnabledLightCount();
        ImGui::Text("Lights: %zu (Enabled: %d)", light_count, enabled_lights);
    }
    
    if (hovered_entity_id_ >= 0) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Hovered: Entity #%d", hovered_entity_id_);
    } else {
        ImGui::Text("Hovered: None");
    }
    
    if (selected_entity_id_ >= 0) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Selected: Entity #%d", selected_entity_id_);
    } else {
        ImGui::Text("Selected: None");
    }
    
    ImGui::Spacing();
    
    ImGui::SeparatorText("Pixel Inspector");
    ImGui::Text("Mouse Position: (%d, %d)", (int)mouse_x_, (int)mouse_y_);
    
    ImGui::Text("Pixel Color:");
    ImGui::SameLine();
    ImGui::ColorButton("##pixel_color_preview", 
                       ImVec4(hovered_pixel_color_.r, hovered_pixel_color_.g, hovered_pixel_color_.b, 1.0f),
                       ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                       ImVec2(40, 20));
    
    ImGui::Text("  R: %.3f", hovered_pixel_color_.r);
    ImGui::Text("  G: %.3f", hovered_pixel_color_.g);
    ImGui::Text("  B: %.3f", hovered_pixel_color_.b);
    
    ImGui::Text("  RGB (8-bit): (%d, %d, %d)", 
                (int)(hovered_pixel_color_.r * 255.0f),
                (int)(hovered_pixel_color_.g * 255.0f),
                (int)(hovered_pixel_color_.b * 255.0f));
    
    size_t total_triangles = 0;
    for (const auto& entity : scene_->GetEntities()) {
        if (entity && entity->GetIndexBuffer()) {
            size_t indices = entity->GetIndexBuffer()->Size() / sizeof(uint32_t);
            total_triangles += indices / 3;
        }
    }
    ImGui::Text("Total Triangles: %zu", total_triangles);

    ImGui::Spacing();

    ImGui::SeparatorText("Render");
    ImGui::Text("Resolution: %d x %d", window_->GetWidth(), window_->GetHeight());
    ImGui::Text("Backend: %s", 
                core_->API() == grassland::graphics::BACKEND_API_VULKAN ? "Vulkan" : "D3D12");
    ImGui::Text("Device: %s", core_->DeviceName().c_str());
    
    ImGui::Spacing();
    
    ImGui::SeparatorText("Accumulation");
    if (!camera_enabled_) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Status: Active");
        ImGui::Text("Samples: %d", film_->GetSampleCount());
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Status: Paused");
        ImGui::Text("(Disable camera to accumulate)");
    }

    ImGui::Spacing();

    ImGui::SeparatorText("Controls");
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Right Click to enable camera");
    ImGui::Text("W/A/S/D - Move camera");
    ImGui::Text("Space/Shift - Up/Down");
    ImGui::Text("Mouse - Look around");
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Hold Tab to hide UI");
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Ctrl+S to save screenshot");

    ImGui::End();
}

void Application::RenderEntityPanel() {
    if (camera_enabled_ || ui_hidden_) {
        return;
    }

    ImGui::SetNextWindowPos(ImVec2((float)window_->GetWidth() - 350.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(350.0f, (float)window_->GetHeight()), ImGuiCond_Always);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | 
                                     ImGuiWindowFlags_NoResize | 
                                     ImGuiWindowFlags_NoCollapse;
    
    if (!ImGui::Begin("Entity Inspector", nullptr, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::SeparatorText("Entity Selection");
    
    const auto& entities = scene_->GetEntities();
    size_t entity_count = entities.size();
    
    ImGui::Text("Select Entity:");
    
    std::string preview_text = selected_entity_id_ >= 0 ? 
        "Entity #" + std::to_string(selected_entity_id_) : 
        "None";
    
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##entity_select", preview_text.c_str())) {
        bool is_selected = (selected_entity_id_ == -1);
        if (ImGui::Selectable("None", is_selected)) {
            selected_entity_id_ = -1;
        }
        if (is_selected) {
            ImGui::SetItemDefaultFocus();
        }
        
        for (size_t i = 0; i < entity_count; i++) {
            std::string label = "Entity #" + std::to_string(i);
            bool is_entity_selected = (selected_entity_id_ == (int)i);
            
            if (ImGui::Selectable(label.c_str(), is_entity_selected)) {
                selected_entity_id_ = (int)i;
            }
            
            if (is_entity_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        
        ImGui::EndCombo();
    }
    
    ImGui::Spacing();
    
    if (selected_entity_id_ >= 0 && selected_entity_id_ < (int)entity_count) {
        ImGui::SeparatorText("Entity Details");
        
        const auto& entity = entities[selected_entity_id_];
        
        ImGui::Text("Transform:");
        glm::mat4 transform = entity->GetTransform();
        glm::vec3 position = glm::vec3(transform[3]);
        ImGui::Text("  Position: (%.2f, %.2f, %.2f)", position.x, position.y, position.z);
        
        glm::vec3 scale = glm::vec3(
            glm::length(glm::vec3(transform[0])),
            glm::length(glm::vec3(transform[1])),
            glm::length(glm::vec3(transform[2]))
        );
        ImGui::Text("  Scale: (%.2f, %.2f, %.2f)", scale.x, scale.y, scale.z);
        
        ImGui::Spacing();
        
        ImGui::SeparatorText("Material");
        Material mat = entity->GetMaterial();
        
        ImGui::Text("Base Color:");
        ImGui::ColorEdit3("##base_color", &mat.base_color[0], ImGuiColorEditFlags_NoInputs);
        ImGui::Text("  RGB: (%.2f, %.2f, %.2f)", mat.base_color.r, mat.base_color.g, mat.base_color.b);
        
        ImGui::Text("Roughness: %.2f", mat.roughness);
        ImGui::Text("Metallic: %.2f", mat.metallic);
        
        ImGui::Spacing();
        
        ImGui::SeparatorText("Mesh");
        if (entity->GetIndexBuffer()) {
            size_t index_count = entity->GetIndexBuffer()->Size() / sizeof(uint32_t);
            size_t triangle_count = index_count / 3;
            ImGui::Text("Triangles: %zu", triangle_count);
            ImGui::Text("Indices: %zu", index_count);
        }
        
        if (entity->GetVertexInfoBuffer()) {
            size_t vertex_size = sizeof(float) * 3;
            size_t vertex_count = entity->GetVertexInfoBuffer()->Size() / vertex_size;
            ImGui::Text("Vertices: %zu", vertex_count);
        }
        
        ImGui::Spacing();
        
        ImGui::SeparatorText("Acceleration Structure");
        if (entity->GetBLAS()) {
            ImGui::Text("BLAS: Built");
        } else {
            ImGui::Text("BLAS: Not built");
        }
    } else {
        ImGui::TextDisabled("No entity selected");
        ImGui::Spacing();
        ImGui::TextWrapped("Hover over an entity to highlight it, then left-click to select. Or use the dropdown above.");
    }
    
    ImGui::End();
}

void Application::RenderLightPanel() {
    if (camera_enabled_ || ui_hidden_ || !light_manager_) {
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(350.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(350.0f, (float)window_->GetHeight() * 0.7f), ImGuiCond_Always);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | 
                                     ImGuiWindowFlags_NoResize | 
                                     ImGuiWindowFlags_NoCollapse;
    
    if (!ImGui::Begin("Light Manager", nullptr, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::SeparatorText("Light Configuration");
    
    auto& lights = light_manager_->GetLights();
    size_t light_count = lights.size();
    
    for (size_t i = 0; i < light_count; i++) {
        Light& light = lights[i];
        
        std::string light_label = "Light " + std::to_string(i);
        std::string type_str;
        switch (light.type) {
            case 0: type_str = "Point Light"; break;
            case 1: type_str = "Area Light"; break;
            case 2: type_str = "Directional Light"; break;
            default: type_str = "Unknown"; break;
        }
        
        if (ImGui::TreeNode((light_label + " - " + type_str).c_str())) {
            // Enabled toggle
            ImGui::Checkbox("Enabled", &light.enabled);
            
            // Light type selector
            const char* light_types[] = { "Point Light", "Area Light", "Directional Light" };
            ImGui::Combo("Type", &light.type, light_types, 3);
            
            // Position (for point and area lights)
            if (light.type != 2) { // Not directional light
                ImGui::DragFloat3("Position", &light.position[0], 0.1f);
            } else {
                // Directional light has position for reference
                ImGui::DragFloat3("Reference Position", &light.position[0], 0.1f);
            }
            
            // Direction (for area and directional lights)
            if (light.type != 0) { // Not point light
                ImGui::DragFloat3("Direction", &light.direction[0], 0.01f, -1.0f, 1.0f);
                
                if (light.type == 1) { // Area light
                    ImGui::DragFloat2("Size", &light.size[0], 0.1f, 0.1f, 10.0f);
                }
                
                if (light.type == 2) { // Directional light
                    ImGui::DragFloat("Cone Angle", &light.cone_angle, 0.5f, 0.0f, 180.0f);
                }
            }
            
            // Color and intensity
            ImGui::ColorEdit3("Color", &light.color[0]);
            ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 10.0f);
            
            ImGui::TreePop();
        }
    }
    
    // Add light button
    if (ImGui::Button("Add Light")) {
        Light new_light = Light::CreatePointLight(glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(1.0f), 1.0f);
        light_manager_->AddLight(new_light);
    }
    
    // Update buffers button
    if (ImGui::Button("Update Light Buffers")) {
        light_manager_->UpdateBuffers();
    }
    
    ImGui::End();
}

void Application::OnRender() {
    if (!alive_) {
        return;
    }

    std::unique_ptr<grassland::graphics::CommandContext> command_context;
    core_->CreateCommandContext(&command_context);
    command_context->CmdClearImage(color_image_.get(), { {0.6, 0.7, 0.8, 1.0} });
    
    command_context->CmdClearImage(entity_id_image_.get(), { {-1, 0, 0, 0} });
    
    command_context->CmdBindRayTracingProgram(program_.get());
    command_context->CmdBindResources(0, scene_->GetTLAS(), grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(1, { color_image_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(2, { camera_object_buffer_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(3, { scene_->GetMaterialsBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(4, { hover_info_buffer_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(5, { entity_id_image_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(6, { film_->GetAccumulatedColorImage() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(7, { film_->GetAccumulatedSamplesImage() }, grassland::graphics::BIND_POINT_RAYTRACING);
    
    // Geometry data
    command_context->CmdBindResources(8, { scene_->GetGeometryDescriptorsBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(9, { scene_->GetVertexInfoBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(10, { scene_->GetIndexBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);

    // Render settings
    command_context->CmdBindResources(11, { render_settings_buffer_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    
    // Light data
    command_context->CmdBindResources(12, { light_manager_->GetLightsBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(13, { light_manager_->GetLightCountBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);
    
    command_context->CmdDispatchRays(window_->GetWidth(), window_->GetHeight(), 1);
    
    grassland::graphics::Image* display_image = color_image_.get();
    if (!camera_enabled_) {
        film_->IncrementSampleCount();
        film_->DevelopToOutput();
        display_image = film_->GetOutputImage();
    }
    
    if (hovered_entity_id_ >= 0 && !camera_enabled_) {
        ApplyHoverHighlight(display_image);
    }
    
    window_->BeginImGuiFrame();
    RenderInfoOverlay();
    RenderEntityPanel();
    RenderLightPanel();
    window_->EndImGuiFrame();
    
    command_context->CmdPresent(window_.get(), display_image);
    core_->SubmitCommandContext(command_context.get());
}