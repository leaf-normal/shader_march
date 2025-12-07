#pragma once
#include "long_march.h"
#include "Scene.h"
#include "Film.h"
#include "Light.h"
#include <memory>

struct CameraObject {
    glm::mat4 screen_to_camera;
    glm::mat4 camera_to_world;
    
    // 景深参数
    float focal_distance;     // 焦点距离（世界空间）
    float aperture_size;      // 光圈直径（控制模糊强度）
    float focal_length;       // 焦距（控制视角）
    float lens_radius;        // 透镜半径 = aperture_size/2
    int enable_depth_of_field;// 是否启用景深效果
};

struct RenderSettings {  // Mainly for random seed
    uint32_t frame_count;
    uint32_t samples_per_pixel;
    uint32_t max_depth;
    uint32_t enable_accumulation;
};

class Application {
public:
    Application(grassland::graphics::BackendAPI api = grassland::graphics::BACKEND_API_DEFAULT);

    ~Application();

    void OnInit();
    void OnClose();
    void OnUpdate();
    void OnRender();
    void UpdateHoveredEntity(); // Update which entity the mouse is hovering over
    void RenderEntityPanel(); // Render entity inspector panel on the right
    void RenderLightPanel(); 

    bool IsAlive() const {
        return alive_;
    }

private:
    // Core graphics objects
    std::shared_ptr<grassland::graphics::Core> core_;
    std::unique_ptr<grassland::graphics::Window> window_;

    // Scene management
    std::unique_ptr<Scene> scene_;
    
    // Film for accumulation
    std::unique_ptr<Film> film_;

    // Camera
    std::unique_ptr<grassland::graphics::Buffer> camera_object_buffer_;
    
    // Hover info buffer
    struct HoverInfo {
        int hovered_entity_id;
    };
    std::unique_ptr<grassland::graphics::Buffer> hover_info_buffer_;

    // Shaders
    std::unique_ptr<grassland::graphics::Shader> raygen_shader_;
    std::unique_ptr<grassland::graphics::Shader> miss_shader_;
    std::unique_ptr<grassland::graphics::Shader> closest_hit_shader_;

    // Rendering
    std::unique_ptr<grassland::graphics::Image> color_image_;
    std::unique_ptr<grassland::graphics::Image> entity_id_image_; // Entity ID buffer for accurate picking
    std::unique_ptr<grassland::graphics::RayTracingProgram> program_;
    bool alive_{ false };

    void ProcessInput(); // Helper function for keyboard input

    glm::vec3 camera_pos_;
    glm::vec3 camera_front_;
    glm::vec3 camera_up_;
    float camera_speed_;

    void OnMouseMove(double xpos, double ypos); // Mouse event handler
    void OnMouseButton(int button, int action, int mods, double xpos, double ypos); // Mouse button event handler
    void RenderInfoOverlay(); // Render the info overlay
    void ApplyHoverHighlight(grassland::graphics::Image* image); // Apply hover highlighting as post-process
    void SaveAccumulatedOutput(const std::string& filename); // Save accumulated output to PNG file

    float yaw_;
    float pitch_;
    float last_x_;
    float last_y_;
    float mouse_sensitivity_;
    bool first_mouse_; // Prevents camera jump on first mouse input
    bool camera_enabled_; // Whether camera movement is enabled
    bool last_camera_enabled_; // Track camera state changes to reset accumulation
    bool ui_hidden_; // Whether UI panels are hidden (Tab key toggle)
    
    // Mouse hovering
    double mouse_x_;
    double mouse_y_;
    int hovered_entity_id_; // -1 if no entity hovered
    glm::vec4 hovered_pixel_color_; // Color value at hovered pixel
    
    // Entity selection
    int selected_entity_id_; // -1 if no entity selected

    uint32_t frame_count_;
    uint32_t samples_per_pixel_;
    std::unique_ptr<grassland::graphics::Buffer> render_settings_buffer_;
    
    std::unique_ptr<LightManager> light_manager_;

    // Depth field parameter
    float focal_distance_;     // 焦点距离
    float aperture_size_;      // 光圈大小
    float focal_length_;       // 焦距
    bool depth_of_field_enabled_; // 是否启用景深
    bool depth_of_field_ui_open_; // 景深设置面板是否打开
    
    float temp_focal_distance_;
    float temp_aperture_size_;
    float temp_focal_length_;
    
    // 应用临时参数到实际参数
    void ApplyDepthOfFieldParams();

    // mouse_scroll
    void OnScroll(double xoffset, double yoffset);
    
    double wheel_accumulator_;
    bool wheel_processed_;
};