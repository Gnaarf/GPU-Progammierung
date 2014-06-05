// geometry shader for growing hair

#version 150

#define OUT_VERTS 14

const float normal_size = 0.01;
const float hair_drop = 0.003;
const float hair_thinning = 0.05;

layout(triangles) in;
in vec3 normal[];
layout(triangle_strip, max_vertices = OUT_VERTS) out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

out vec4 geomColor;

void main(void)
{
	geomColor = vec4(1, 0, 0, 1);
	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		//vec4 offset = vec4(0.01,0,0,0);
		vec3 p = gl_in[i].gl_Position.xyz;
		vec3 n = normalize(normal[i].xyz);
		vec3 camDir = vec3(View[0][2],View[1][2],View[2][2]);
		vec3 offsetVec = cross(n,camDir);
		offsetVec = normalize(offsetVec);

		//gl_Position = gl_in[i].gl_Position;
		gl_Position = Projection * View * (vec4(p,1) - 0.01*vec4(offsetVec,0));
		EmitVertex();
		gl_Position = Projection * View * (vec4(p,1) + 0.01*vec4(offsetVec,0));
		EmitVertex();
		vec4 c = vec4(p,1);
		//c = c + normal_size*vec4(n,0);
		//gl_Position = Projection * View * c;
		//EmitVertex();
		for(int j = 1; j <= OUT_VERTS/2; ++j)
		{
			vec4 diff = normal_size*vec4(n,0) - hair_drop*j*vec4(0,1,0,0);
			diff = normal_size*normalize(diff);
			c = c + diff;
			//offsetVec = cross(camDir,diff.xyz);
			//offsetVec = normalize(offsetVec);
			gl_Position = Projection * View * (c - (OUT_VERTS - j) * hair_thinning * 0.01*vec4(offsetVec,0));

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
			gl_Position = Projection * View * (c + (OUT_VERTS - j) * hair_thinning * 0.01*vec4(offsetVec,0));
			EmitVertex();
		}
		EndPrimitive();
	}
}
