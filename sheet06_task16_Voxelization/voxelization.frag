// Extension aktivieren, damit << im Befehlssatz vorliegt.
#extension GL_EXT_gpu_shader4 : enable

// Ausgabevariable
varying out uvec4 result;

void main()
{	
	// TODO: Tiefenwert von [0..1] auf {0..127} abbilden.
	int bitPos = int(127 * gl_FragCoord.z);
	ivec4 outgoing = ivec4(0,0,0,0);

	if( bitPos < 32)
	{
		//outgoing.w = 1 << bitPos;
		int tmp = 1 << bitPos;
		outgoing.w = tmp - 1 + tmp;
		outgoing.z = 0;
		outgoing.y = 0;
		outgoing.x = 0;
		//result.x = 1 << 31;
	}
	else if(bitPos < 64)
	{
		int tmp = 1 << (bitPos - 32);
		//outgoing.z = 1 << (bitPos - 32);
		outgoing.w = 4294967295;
		outgoing.z = tmp -1 + tmp;
		outgoing.y = 0;
		outgoing.x = 0;
		//result.y = 1 << 31;
	}
		else if(bitPos < 96)
	{
		//outgoing.w = 1 << (bitPos - 64);
		
		int tmp = 1 << (bitPos - 64);
		outgoing.w = 4294967295;
		outgoing.z = 4294967295;
		outgoing.y = tmp -1 + tmp;
		outgoing.x = 0;
		//result.z = 1 << 31;
	}
	else
	{
		//outgoing.x = 1 << (bitPos - 96);
		int tmp = 1 << (bitPos - 96);
		outgoing.w = 4294967295;
		outgoing.z = 4294967295;
		outgoing.y = 4294967295;
		outgoing.x = tmp -1 + tmp;
	}
	
	result = uvec4(outgoing);
	
	// Dies ergibt beispielsweise den Wert 42.
	// Erzeugen Sie nun eine bit-Maske, in der das (im Beispiel) 42te Bit (von rechts gezählt) eine 1 ist und alle anderen eine 0.
	// 00000000..000000010000000..00000000
	// |<- 86 Nullen ->| |<- 41 Nullen ->|
	//                  ^
	//                Bit 42
	// Weisen Sie diese bit-Maske der Variable 'result' zu.
}
