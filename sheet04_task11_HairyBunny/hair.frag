// simple fragment shader that outputs transparent white (as hair color)

#version 150
in vec4 geomColor;
out vec4 fragColor;

void main()
{		
	fragColor = /*vec4(0.75, 0.375, 0.075, 1)*/ geomColor;
}
