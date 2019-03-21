#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <chrono>
#include <iostream>
#include <map>
#include <cmath>
#include <stdlib.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void renderQuad();
void renderCube();

const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720 ;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// buffers
int BUF_WIDTH;
int BUF_HEIGHT;

int main()  {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // GLFW window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "scene", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetFramebufferSize(window, &BUF_WIDTH, &BUF_HEIGHT);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader shaderGeometryPass("g_buffer.vert", "g_buffer.frag");
    Shader shaderLightingPass("deferred_shading.vert", "deferred_shading.frag");
    Shader shaderLightBox("deferred_light_box.vert", "deferred_light_box.frag");

    // Model boat(FileSystem::getPath("assets/boat/boat.obj"));
    // Model nanosuit(FileSystem::getPath("assets/nanosuit/nanosuit.obj"));
    Model earth(FileSystem::getPath("assets/sphere/sphere.obj"), "earth");
    Model mars(FileSystem::getPath("assets/mars/sphere.obj"), "mars");
    Model mercury(FileSystem::getPath("assets/mercury/sphere.obj"), "mercury");
    Model venus(FileSystem::getPath("assets/venus/sphere.obj"), "venus");
    Model jupiter(FileSystem::getPath("assets/jupiter/sphere.obj"), "jupiter");
    // Model floor(FileSystem::getPath("assets/plane/plane.obj"), "plane");
    // Model cornell_box(FileSystem::getPath("assets/cornell_box/cornell_box.obj"));
    // Model miami(FileSystem::getPath("assets/miami/miami.obj"));
    // Model miami(FileSystem::getPath("assets/miami/miami.obj"));
    // Model Ocean(FileSystem::getPath("assets/Ocean/Ocean.obj"));
    // Model poterie_obj(FileSystem::getPath("assets/poterie_obj/poterie_obj.obj"));
//    std::vector<glm::vec3> objectPositions;
//    objectPositions.emplace_back(glm::vec3(-3.0,  -3.0, -3.0));
//    objectPositions.emplace_back(glm::vec3( 0.0,  -3.0, -3.0));

    std::map<Model, glm::vec3> objs;
    // objs.insert(std::pair<Model, glm::vec3>(floor, glm::vec3(0.0, -3.0, 0.0)));
    objs.insert(std::pair<Model, glm::vec3>(earth, glm::vec3(0.0, 0.0, 0.0)));
    objs.insert(std::pair<Model, glm::vec3>(mars, glm::vec3(-4.0, 0.0, 6.0)));
    objs.insert(std::pair<Model, glm::vec3>(venus, glm::vec3(5.0, 0.0, 0.0)));
    objs.insert(std::pair<Model, glm::vec3>(mercury, glm::vec3(7.0, 0.0, 0.0)));
    objs.insert(std::pair<Model, glm::vec3>(jupiter, glm::vec3(-11.0, 0.0, 6.0)));
    // objs.insert(std::pair<Model, glm::vec3>(boat, glm::vec3(-3.0,  -3.0, -3.0)));
    // objs.insert(std::pair<Model, glm::vec3>(nanosuit, glm::vec3(0.0,  -3.0, -3.0)));
    // objs.insert(std::pair<Model, glm::vec3>(poterie_obj, glm::vec3(-3.0, -3.0, 0.0)));
//    objs.insert(std::pair<Model, glm::vec3>(sphere, glm::))


    // configure g-buffer framebuffer
    // ------------------------------
    unsigned int gBuffer;
    glGenFramebuffers(1, &gBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    unsigned int gPosition, gNormal, gAlbedoSpec;

    // position color buffer
    glGenTextures(1, &gPosition);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, BUF_WIDTH, BUF_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);

    // normal color buffer
    glGenTextures(1, &gNormal);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, BUF_WIDTH, BUF_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);

    // color + specular color buffer
    glGenTextures(1, &gAlbedoSpec);
    glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, BUF_WIDTH, BUF_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0);

    // tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, attachments);

    // create and attach depth buffer (renderbuffer)
    unsigned int rboDepth;
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, BUF_WIDTH, BUF_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

    // create a fbo for the depth map for the light.
//    unsigned int depthMapFBO;
//    glGenFramebuffers(1, &depthMapFBO);
//
//    const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
//    unsigned int depthMap;
//    glGenTextures(1, &depthMap);
//    glBindTexture(GL_TEXTURE_2D, depthMap);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
//                 SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//
//    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
//    glDrawBuffer(GL_NONE);
//    glReadBuffer(GL_NONE);
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    // finally check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // lighting info
    // -------------
    const unsigned int NR_LIGHTS = 1;
    std::vector<glm::vec3> lightPositions;
    std::vector<glm::vec3> lightColors;
    std::vector<glm::vec3> lightDir;
//    srand(13);


    lightPositions.emplace_back(glm::vec3(0.0f, 2.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(2.0f, 2.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.5f, 0.5f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(10.0f, 10.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.8f, 0.8f, 0.8f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(8.0f, 8.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.7f, 0.5f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(5.0f, 10.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.7f, 0.6f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(8.0f, 9.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.5f, 0.9f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(-10.0f, 9.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.8f, 0.8f, 0.8f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(-8.0f, 9.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.7f, 0.5f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(-5.0f, 8.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.7f, 0.6f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));

    lightPositions.emplace_back(glm::vec3(-8.0f, 7.0f, 0.0f));
    lightColors.emplace_back(glm::vec3(0.5f, 0.9f, 1.0f));
    lightDir.emplace_back(glm::vec3(-0.0f, -1.0f, -0.0f));


//    for (unsigned int i = 0; i < NR_LIGHTS; i++)
//    {
//        // calculate slightly random offsets
//        float xPos = ((rand() % 100) / 100.0f) * 6.0f - 3.0f;
//        float yPos = ((rand() % 100) / 100.0f) * 6.0f - 1.0f;
//        float zPos = ((rand() % 100) / 100.0f) * 6.0f - 3.0f;
//        lightPositions.emplace_back(glm::vec3(xPos, yPos, zPos));
//
//        // also calculate random color
//        float rColor = ((rand() % 100) / 100.0f); // between 0.5 and 1.0
//        float gColor = ((rand() % 100) / 100.0f); // between 0.5 and 1.0
//        float bColor = ((rand() % 100) / 100.0f); // between 0.5 and 1.0
//        lightColors.emplace_back(glm::vec3(rColor, gColor, bColor));
//    }

    // shader configuration
    // --------------------
    shaderLightingPass.use();
    shaderLightingPass.setInt("gPosition", 0);
    shaderLightingPass.setInt("gNormal", 1);
    shaderLightingPass.setInt("gAlbedoSpec", 2);

    // render loop
    // -----------
    bool once = true;
    std::clock_t start;
    float duration;
    start = std::clock();
    int count = 0;
    bool subtract = true;
    while (!glfwWindowShouldClose(window))  {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
//        std::cout << "FPS: " << 1./deltaTime << std::endl;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 1. geometry pass: render scene's geometry/color data into gbuffer
        // -----------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)BUF_WIDTH / (float)BUF_HEIGHT, 0.1f, 1000.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        shaderGeometryPass.use();
        shaderGeometryPass.setMat4("projection", projection);
        shaderGeometryPass.setMat4("view", view);

        duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;;

        auto i = 0;
        for (auto &pair : objs)   {
            model = glm::mat4(1.0f);
            model = glm::translate(model, pair.second);
//            if (i == 2)
//                model = glm::rotate(model, 3.1415f/2.f, glm::vec3(1, 0, 0));
//            else if (i == 3)
//                model = glm::rotate(model, 3.1415f/2.f, glm::vec3(-1, 0, 0));
//            else if (i == 4)
//                model = glm::rotate(model, 3.1415f/2.f, glm::vec3(0, 0, -1));
//            else if (i == 5)
//                model = glm::rotate(model, 3.1415f/2.f, glm::vec3(0, 0, 1));

            if (pair.first.name == "mars") {
                model = glm::scale(model, glm::vec3(0.075f));
            } else if (pair.first.name == "mercury") {
                model = glm::scale(model, glm::vec3(0.025f));
            } else if (pair.first.name == "venus") {
                model = glm::scale(model, glm::vec3(0.05f));
            } else if (pair.first.name == "jupiter") {
                model = glm::scale(model, glm::vec3(0.15f));
            } else {
                model = glm::scale(model, glm::vec3(0.125f));
            }
            shaderGeometryPass.setMat4("model", model);
            Model toDraw = pair.first;
            toDraw.Draw(shaderGeometryPass);
            ++i;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // 2. lighting pass: calculate lighting by iterating over a screen filled quad pixel-by-pixel using the gbuffer's content.
        // -----------------------------------------------------------------------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaderLightingPass.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
        // send light relevant uniforms
        shaderLightingPass.setVec3("cameraPos", camera.Position);
        shaderLightingPass.setMat4("cameraRot", camera.GetViewMatrix());
        shaderLightingPass.setMat4("invCameraRot", glm::inverse(camera.GetViewMatrix()));
        shaderLightingPass.setFloat("seed", 10.f);
        for (unsigned int i = 0; i < lightPositions.size(); i++)
        {
            shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].position", lightPositions[i]);
            shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].color", lightColors[i]);
            shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].direction", lightDir[i]);
            // update attenuation parameters and calculate radius
            const float constant = 1.0; // note that we don't send this to the shader, we assume it is always 1.0 (in our case)
            const float linear = 0.000045;
            const float quadratic = 0.075;
            shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].linear", linear);
            shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].quadratic", quadratic);
            // then calculate radius of light volume/sphere
            const float maxBrightness = std::fmaxf(std::fmaxf(lightColors[i].r, lightColors[i].g), lightColors[i].b);
            float radius = (-linear + std::sqrt(linear * linear - 4 * quadratic * (constant - (256.0f / 5.0f) * maxBrightness))) / (2.0f * quadratic);
            shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].radius", radius);
            shaderLightingPass.setFloat("lights["+std::to_string(i) + "].physicalRadius", 0.4f);

            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].intensity", 2.0f + 0.5f * std::sin( 1000 * M_PI * duration / 3) * r );
        }
        shaderLightingPass.setVec3("viewPos", camera.Position);
        shaderLightingPass.setFloat("fov", glm::radians(camera.Zoom));
        shaderLightingPass.setVec2("u_resolution", glm::vec2(BUF_WIDTH, BUF_HEIGHT));
        shaderLightingPass.setVec3("cam.eye", camera.Position);
        shaderLightingPass.setVec3("cam.front", camera.Front);
        shaderLightingPass.setVec3("cam.up", camera.Up);
        shaderLightingPass.setFloat("cam.fov", camera.Zoom);
        shaderLightingPass.setMat4("projection", projection);
        shaderLightingPass.setVec3("spheres[0].position", objs.at(earth));
        shaderLightingPass.setFloat("spheres[0].radius", .125f * 20.f);

        
        shaderLightingPass.setFloat("time", duration);

        lightPositions[0] = glm::vec3(0.0f + 5.0 * std::sin(5 * M_PI * duration / 3), 2.0f, 0.0f + 5.0 * std::sin(2 * M_PI * duration / 3));
        lightPositions[1] = glm::vec3(0.0f - 5.0 * std::sin(5 * M_PI * duration / 3), 2.0f, 0.0f - 5.0 * std::sin(2 * M_PI * duration / 3));


        // finally render quad
        renderQuad();

        // 2.5. copy content of geometry's depth buffer to default framebuffer's depth buffer
        // ----------------------------------------------------------------------------------
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
        // blit to default framebuffer. Note that this may or may not work as the internal formats of both the FBO and default framebuffer have to match.
        // the internal formats are implementation defined. This works on all of my systems, but if it doesn't on yours you'll likely have to write to the
        // depth buffer in another shader stage (or somehow see to match the default framebuffer's internal format with the FBO's internal format).
        glBlitFramebuffer(0, 0, BUF_WIDTH, BUF_HEIGHT, 0, 0, BUF_WIDTH, BUF_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // 3. render lights on top of scene
        // --------------------------------
//        shaderLightBox.use();
//        shaderLightBox.setMat4("projection", projection);
//        shaderLightBox.setMat4("view", view);
//        for (unsigned int i = 0; i < lightPositions.size(); i++)
//        {
//            model = glm::mat4(1.0f);
//            model = glm::translate(model, lightPositions[i]);
//            model = glm::scale(model, glm::vec3(0.01f));
//            shaderLightBox.setMat4("model", model);
//            shaderLightBox.setVec3("lightColor", lightColors[i]);
//            sphere.Draw(shaderLightBox);
//        }


        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        if (once) {
            glfwSetWindowPos(window, 100, 100);
            once = false;
        }
    }

    glfwTerminate();
    return 0;
}

// renderCube() renders a 1x1 3D cube in NDC.
// -------------------------------------------------
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
void renderCube()
{
    // initialize (if necessary)
    if (cubeVAO == 0)
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right
            1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
            // front face
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
            1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            // left face
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            // right face
            1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
            1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
            1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right
            1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
            1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
            1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left
            // bottom face
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
            1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            // top face
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right
            1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left
        };
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        // fill buffer
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // link vertex attributes
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    // render Cube
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}


// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
            1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)   {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            camera.ProcessKeyboard(FORWARD, deltaTime * 5.0f);
        else
            camera.ProcessKeyboard(FORWARD, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            camera.ProcessKeyboard(BACKWARD, deltaTime * 5.0f);
        else
            camera.ProcessKeyboard(BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime * 5.0f);
        else
            camera.ProcessKeyboard(LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime * 5.0f);
        else
            camera.ProcessKeyboard(RIGHT, deltaTime);    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)   {
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)   {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)    {
    camera.ProcessMouseScroll(yoffset);
}
