#include "app.h"

int main() {
  // Create only one application instance to avoid ImGui conflicts
  // Change BACKEND_API_D3D12 to BACKEND_API_VULKAN if you prefer Vulkan
  Application app{grassland::graphics::BACKEND_API_VULKAN};

  app.OnInit();

  while (app.IsAlive()) {
    app.OnUpdate();
    if (app.IsAlive()) {  // Check again after update
      app.OnRender();
    }
    glfwPollEvents();
  }

  app.OnClose();

  return 0;
}
