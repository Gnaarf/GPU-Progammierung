// Extension aktivieren, damit << im Befehlssatz vorliegt.
#extension GL_EXT_gpu_shader4 : enable

// Ausgabevariable
varying out uvec4 result;

void main()
{	
	// TODO: Tiefenwert von [0..1] auf {0..127} abbilden.

	uint bitPos = uint(127 * gl_FragCoord.z);

	uint foo = uint(1);

	uvec4 tmp = uvec4(0);
	tmp.x = foo << 30;
	
	// Dies ergibt beispielsweise den Wert 42.
	// Erzeugen Sie nun eine bit-Maske, in der das (im Beispiel) 42te Bit (von rechts gez�hlt) eine 1 ist und alle anderen eine 0.
	// 00000000..000000010000000..00000000
	// |<- 86 Nullen ->| |<- 41 Nullen ->|
	//                  ^
	//                Bit 42
	// Weisen Sie diese bit-Maske der Variable 'result' zu.
	result = tmp;
}
