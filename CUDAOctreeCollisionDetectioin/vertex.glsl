#version 330 core

layout(location = 0) in vec3 pos;

uniform mat4 m;
uniform mat4 v;
uniform mat4 p;

uniform vec3 color;

out vec3 fragColor;

void main()
{
    gl_Position = p * v * m * vec4(pos.x, pos.y, pos.z, 1.0);
    fragColor = color;
}