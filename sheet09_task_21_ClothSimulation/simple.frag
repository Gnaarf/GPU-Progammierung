
#version 330

in vec2 out_TexCoord;
// now interpolated
in vec3 vertexShaderOut_Normal;

out vec4 FragColor;

void main()
{	
	vec4 color = vec4(1);
	vec3 hue = vec3(out_TexCoord * max(length(dFdx(out_TexCoord)), length(dFdy(out_TexCoord))) * 70, 0.5);
	
	float diffuseLightCoefficient = max(0, dot(normalize(vertexShaderOut_Normal), vec3(0, 0, -1)));

	//color.rgb = abs(vertexShaderOut_Normal);
	//color.rgb = vec3(1);
	//color.rgb = hue * diffuseLightCoefficient;
	color.rgb = vec3(diffuseLightCoefficient);

	FragColor = color;
}
