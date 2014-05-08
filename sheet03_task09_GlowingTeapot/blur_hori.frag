uniform sampler2D texture;

// Filtergroesse (gesamt)
uniform int filterWidth;

// Hier soll der Filter implementiert werden
void main()
{
        // Schrittweite fuer ein Pixel (bei Aufloesung 512)
        float texCoordDelta = 1. / 512.;

        // linke Ecke des Filters
        vec2 texCoord;
        texCoord.x = gl_TexCoord[0].s - (float(filterWidth / 2) * texCoordDelta);
        texCoord.y = gl_TexCoord[0].t;

        // Wert zum Aufakkumulieren der Farbwerte
        vec3 val = vec3(0);

        vec2 texelDelta = vec2(0);

        for(int dx = 0; dx < filterWidth; dx++)
        {
                texelDelta.x = float(dx);
                val = val + texture2D(texture, texCoord + texelDelta*texCoordDelta).rgb;
        }

        // Durch filterWidth teilen, um zu normieren.
        val = 2.0 * val / float(filterWidth);

        // TODO: Ausgabe von val
        gl_FragColor.rgb = val;
        gl_FragColor.a = 1.0f;

        // Die folgende Zeile dient nur zu Debugzwecken!
        // Wenn das Framebuffer object richtig eingestellt wurde und die Textur an diesen Shader übergeben wurde
        // wird die Textur duch den folgenden Befehl einfach nur angezeigt.
        //gl_FragColor.rgb = texture2D(texture,gl_TexCoord[0].st).xyz;
}
