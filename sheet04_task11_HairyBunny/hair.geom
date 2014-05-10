// geometry shader for growing hair

#version 150

#define OUT_VERTS 9

const float normal_size = 0.01;
const float hair_drop = 0.003;

layout(triangles) in;
in vec3 normal[];
layout(line_strip, max_vertices = OUT_VERTS) out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

out vec4 geomColor;

void main(void)
{
	geomColor = vec4(1, 0, 0, 1);
	//Pass-thru!
	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		vec3 p = gl_in[i].gl_Position.xyz;
		vec3 n = normalize(normal[i].xyz);

		//gl_Position = gl_in[i].gl_Position;
		gl_Position = Projection * View * vec4(p,1);
		EmitVertex();
		vec4 c = vec4(p,1);
		c = c + normal_size*vec4(n,0);
		gl_Position = Projection * View * c;
		EmitVertex();
		for(int j = 1; j <= OUT_VERTS; ++j)
		{
			vec4 diff = normal_size*vec4(n,0) - hair_drop*j*vec4(0,1,0,0);
			diff = normal_size*normalize(diff);
			c = c + diff;
			gl_Position = Projection * View * c;

			switch(j)
			{
				case 1:
					geomColor = vec4(0,1,0,1);
					break;
				case 2:
					geomColor = vec4(0,0,1,1);
					break;
				case 3:
					geomColor = vec4(1,1,0,1);
					break;
				case 4:
					geomColor = vec4(0,1,1,1);
					break;
				case 5:
					geomColor = vec4(0.5,0,0,1);
					break;
				case 6:
					geomColor = vec4(0,0.5,0,1);
					break;
				case 7:
					geomColor = vec4(0,0,0.5,1);
					break;
				case 8:
					geomColor = vec4(0.5,0.5,0,1);
					break;
				case 9:
					geomColor = vec4(0.5,0,0.5,1);
					break;
			}

			EmitVertex();
		}
		EndPrimitive();
	}
}
