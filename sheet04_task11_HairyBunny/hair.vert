#version 330

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;

out vec3 normal;

void main()
{	
	normal = in_Normal;
	gl_Position = vec4(in_Position,1);
}
