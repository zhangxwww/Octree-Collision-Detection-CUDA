#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include "Shader.h"
#include "KeyboardController.h"
#include "World.h"

#include "Constant.cuh"


World world;
KeyboardController keyboardController;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
GLFWwindow* init();
void initSphere();

std::vector<float> sphereVertices;
std::vector<int> sphereIndices;

int main() {
    GLFWwindow* window = init();
    if (window == nullptr) {
        return 0;
    }

    initSphere();

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), &sphereVertices[0], GL_STATIC_DRAW);

    GLuint element_buffer_object;
    glGenBuffers(1, &element_buffer_object);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(int), &sphereIndices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    unsigned int wallVBO, wallVAO, wallEBO;
    glGenVertexArrays(1, &wallVAO);
    glGenBuffers(1, &wallVBO);
    glGenBuffers(1, &wallEBO);
    glBindVertexArray(wallVAO);
    glBindBuffer(GL_ARRAY_BUFFER, wallVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(wall.vertex), wall.vertex, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wallEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(wall.indices), wall.indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    Shader shader;
    shader.loadFromFile("vertex.glsl", "fragment.glsl");

    world.addBalls(1000);

    float time_until_update = 0;
    float lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window) && !keyboardController.shouldClose()) {
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        shader.use();

        keyboardController.step(deltaTime);
        world.step(deltaTime, time_until_update);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)(WIDTH) / HEIGHT, 1.0f, 100.0f);
        glm::mat4 view = glm::lookAt(keyboardController.getEyePos(), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        shader.setMat4("v", view);
        shader.setMat4("p", projection);

        // draw balls
        for (Ball* b : world.getBalls()) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, b->pos);
            model = glm::scale(model, glm::vec3(b->radius));

            shader.setMat4("m", model);
            shader.setVec3("color", b->color);

            glBindVertexArray(VAO);
            glPointSize(2);
            glDrawElements(GL_POINTS, X_SEGMENTS * Y_SEGMENTS * 6, GL_UNSIGNED_INT, 0);
        }

        // draw walls
        shader.setVec3("color", wall.color);

        shader.setMat4("m", wall.floorTrans);
        glBindVertexArray(wallVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        shader.setMat4("m", wall.ceilTrans);
        glBindVertexArray(wallVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
       
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &element_buffer_object);

    glfwTerminate();
    return 0;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    keyboardController.callback(window, key, scancode, action, mode);
}

GLFWwindow* init() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, false);

    auto window = glfwCreateWindow(WIDTH, HEIGHT, "Collision Detection", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cout << "Failed to Create OpenGL Context" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return nullptr;
    }
    glViewport(0, 0, WIDTH, HEIGHT);
    return window;
}

void initSphere() {

    for (int y = 0; y <= Y_SEGMENTS; y++)
    {
        for (int x = 0; x <= X_SEGMENTS; x++)
        {
            float xSegment = (float)x / (float)X_SEGMENTS;
            float ySegment = (float)y / (float)Y_SEGMENTS;
            float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            float yPos = std::cos(ySegment * PI);
            float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            sphereVertices.push_back(xPos);
            sphereVertices.push_back(yPos);
            sphereVertices.push_back(zPos);
        }
    }

    for (int i = 0; i < Y_SEGMENTS; i++)
    {
        for (int j = 0; j < X_SEGMENTS; j++)
        {
            sphereIndices.push_back(i * (X_SEGMENTS + 1) + j);
            sphereIndices.push_back((i + 1) * (X_SEGMENTS + 1) + j);
            sphereIndices.push_back((i + 1) * (X_SEGMENTS + 1) + j + 1);
            sphereIndices.push_back(i * (X_SEGMENTS + 1) + j);
            sphereIndices.push_back((i + 1) * (X_SEGMENTS + 1) + j + 1);
            sphereIndices.push_back(i * (X_SEGMENTS + 1) + j + 1);
        }
    }
}
