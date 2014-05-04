uniform sampler2D texture;

// Hier soll der Filter implementiert werden
void main()
{
        // Schrittweite fuer ein Pixel (bei Aufloesung 512)
        float texCoordDelta = 1. / 512.;

        // Filtergroesse (gesamt)
        int filterWidth = 8;

        // linke Ecke des Filters
        vec2 texCoord;
        texCoord.x = gl_TexCoord[0].s - (float(filterWidth / 2) * texCoordDelta);
        texCoord.y = gl_TexCoord[0].t - (float(filterWidth / 2) * texCoordDelta);

        // Wert zum Aufakkumulieren der Farbwerte
        vec3 val = vec3(0);

        vec2 texelDelta = vec2(0);

        for(int dy = 0; dy < filterWidth; dy++)
        {
                texelDelta.y = float(dy);
                for(int dx = 0; dx < filterWidth; dx++)
                {
                        texelDelta.x = float(dx);
                        val = val + texture2D(texture, texCoord + texelDelta*texCoordDelta).rgb;

                        //TODO: Verschieben der Texturkoordinate -> naechstes Pixel in x Richtung
                }
                // TODO: Zurücksetzen von texCoord.x und weiterschieben von texCoord.y
        }

        // Durch filterWidth^2 teilen, um zu normieren.
        val = 2.0 * val / float(filterWidth*filterWidth);

        // TODO: Ausgabe von val
        gl_FragColor.rgb = val;
        gl_FragColor.a = 1.0f;

        // Die folgende Zeile dient nur zu Debugzwecken!
        // Wenn das Framebuffer object richtig eingestellt wurde und die Textur an diesen Shader übergeben wurde
        // wird die Textur duch den folgenden Befehl einfach nur angezeigt.
        //gl_FragColor.rgb = texture2D(texture,gl_TexCoord[0].st).xyz;
}
