// uniform highp vec4 diffuse;
//varying color vec4 v_color;
varying vec4 particleColor;
//varying vec3 n;
varying vec2 u;
varying vec2 v;
void main ()
{
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    
    if (
    (dot(circCoord, u)*dot(circCoord, u)+dot(circCoord, v)*dot(circCoord, v))
     > (1.0)) { discard; }

    
    
    //if (dot(circCoord, circCoord) > (1.0)) { discard; }
    gl_FragColor = particleColor;
}