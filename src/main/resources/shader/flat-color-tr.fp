
uniform float zMin;
uniform float zMax;
uniform float zSliceMin;
uniform float zSliceMax;
varying vec4 particleColor;
varying float zPosFrag;
void main ()
{
    float intensity=1.0;
    if ((zPosFrag<zMin)||(zPosFrag>zMax)) discard;
    if ((zPosFrag<zSliceMax)&&(zPosFrag>zSliceMin)) intensity = 0.5;    
    gl_FragColor = intensity*particleColor;    
}