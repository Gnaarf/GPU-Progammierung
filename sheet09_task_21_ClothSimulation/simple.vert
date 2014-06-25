
#version 330

in layout(location=0) vec3 in_Position;
in layout(location=1) vec2 in_TexCoord;

// TODO: in parameter for normals
in layout(location=2) vec3 vertexShaderIn_Normal;

out vec2 out_TexCoord;
out vec3 vertexShaderOut_Normal;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main()
{
	gl_Position = Projection * View * vec4(in_Position, 1);
	out_TexCoord = in_TexCoord;
	//vertexShaderOut_Normal = (View * vec4(vertexShaderIn_Normal, 1)).xyz;
	vertexShaderOut_Normal = (inverse(transpose(View)) * vec4(vertexShaderIn_Normal,0)).xyz;
}
