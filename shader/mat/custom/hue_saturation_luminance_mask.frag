//Hue Saturation Luminance Transforms for Toolbag 4
//A: Chris Perrella

#include "../state.frag"

USE_TEXTURE2D(tMaskMap); //srgb name "Mask Map"
uniform int uMaskMapUseSecondaryUV; //Bool name "Use Secondary UV" default 0 

uniform int uRed; //Bool name "Red Mask" default 1
uniform int uGreen; //Bool name "Green Mask" default 0
uniform int uBlue; //Bool name "Blue Mask" default 0
uniform int uAlpha; //Bool name "Alpha Mask" default 0
uniform int uInvert; //Bool name "Invert Mask" default 0

uniform float uHue; // name "Hue" default 0.5
uniform float uSat; // name "Saturation" default 0.5
uniform float uLum; // name "Luminance" default 0.5

#define HSV_EPSILON 1.0e-10

inline vec3 rgb_hcv(vec3 rgb_color) 
{
    vec4 p = ( rgb_color.g < rgb_color.b ) 
        ? vec4( rgb_color.bg, -1.0, 2.0/3.0 ) 
        : vec4( rgb_color.gb, 0.0, -1.0/3.0 );
    
    vec4 q = ( rgb_color.r < p.x ) 
        ? vec4( p.xyw, rgb_color.r ) 
        : vec4( rgb_color.r, p.yzx );
    
    // Compute the chroma, which is the difference between the max and min 
    // of the red, green, and blue components.
    float chroma = q.x - min( q.w, q.y );
    float hue = abs( ( q.w - q.y ) / ( 6.0 * chroma + HSV_EPSILON ) + q.z );
    
    return vec3( hue, chroma, q.x );
}

inline vec3 rgb_hsv(vec3 rgb)
{
    vec3 hcv = rgb_hcv( rgb );
    float s = hcv.y / ( hcv.z + HSV_EPSILON );
    return vec3( hcv.x, s, hcv.z );
}

inline vec3 hue_rgb(float h)
{
    float red = abs( h * 6.0 - 3.0 ) - 1.0;
    float green = 2.0 - abs( h * 6.0 - 2.0 );
    float blue = 2.0 - abs( h * 6.0 - 4.0 );
    return saturate( vec3( red, green, blue ) );
}

inline vec3 hsv_rgb(vec3 hsv)
{
    vec3 rgb = hue_rgb( hsv.x );
    return ( ( rgb - 1.0 ) * hsv.y + 1.0 ) * hsv.z;
}

inline vec3 adjust_hsv(vec3 color, float hue, float sat, float lum)
{
    vec3 hsv = rgb_hsv(color); 

    float hue_shift = (hue - 0.5) * 2.0;
    hsv.x += hue_shift;
    if(hsv.x < 0.0) hsv.x += 1.0;
    else if(hsv.x > 1.0) hsv.x -= 1.0;

    hsv.y *= ( sat * 2.0 );
    hsv.z *= ( lum * 2.0 );

    return hsv_rgb( hsv );
}

void    AlbedoHueSatLum(inout FragmentState s)
{
    float n = 0.0;

    vec2 uv = lerp( s.vertexTexCoord.xy, s.vertexTexCoordSecondary.xy, uMaskMapUseSecondaryUV );
    vec4 channel =vec4( uRed, uGreen, uBlue, uAlpha );
    float mask_map = ( dot( channel, 1.0 ) > 0.0 ) ? 
                       dot( channel, texture2D( tMaskMap, uv ) ) : 
                       1.0;

    #ifdef Albedo
        Albedo(s);
    #endif

    vec3 adjust_diffuse   = adjust_hsv( s.albedo.xyz, uHue, uSat, uLum );

    s.albedo.xyz = mix( s.albedo.xyz, 
                        adjust_diffuse,
                        ( ( uInvert > 0.0 ) ? 
                            ( 1.0 - mask_map ) : 
                            mask_map )
                    );
}

#ifdef Albedo
    #undef Albedo
#endif
#define Albedo    AlbedoHueSatLum
