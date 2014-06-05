uniform sampler2D imageTexture;

const float SCREEN_WIDTH = 512.0;

void main()
{	
	vec4 position = gl_Vertex;

	// TODO: Farbe auslesen
	vec4 color = texture2D(imageTexture, gl_MultiTexCoord0.st).rgba;
 	
	// TODO: Grauwert berechnen
	float intensity = 0.2126*color.r + 0.7152*color.g + 0.0722*color.b;
	
	// TODO: x-Position berechnen. Das Zielpixel ist zwischen (0,0) und (255,0)
	position.x = floor(intensity*256.0);
	position.y = 0.0;
	
	// TODO: Die Position in [0,1] auf das Intervall [-1,1] abbilden.
	// hier mit Hilfe der entsprechend eingestellten Matrizen
	gl_Position = gl_ModelViewProjectionMatrix*position;
}
