//4 Channel Detail Normal Map Shader for Toolbag 4
//A: Chris Perrella
#include "../state.frag"

USE_TEXTURE2D(tMaskMap); //linear name "RGBA Detail Mask"
uniform int uCustomDetailNormalUseSecondaryUV; //Bool name "Use Secondary UV" default 0 

uniform int uRedEnable; //name "Enable Red Map" bool default 0
USE_TEXTURE2D(tRedDetail); //linear name "Red Detail Normal Map"
uniform vec2 uRedTiling; //name "Red UV Tiling" default 4,4
uniform float uRedDetailWeight; //name "Red Detail Weight" default 1
uniform int uFlipRedY; //name "Flip Red Y" bool default 0

uniform int uGreenEnable; //name "Enable Green Map" bool default 0
USE_TEXTURE2D(tGreenDetail); //linear name "Green Detail Normal Map"
uniform vec2 uGreenTiling; //name "Green UV Tiling" default 4,4
uniform float uGreenDetailWeight; //name "Green Detail Weight" default 1
uniform int uFlipGreenY; //name "Flip Green Y" bool default 0

uniform int uBlueEnable; //name "Enable Blue Map" bool default 0
USE_TEXTURE2D(tBlueDetail); //linear name "Blue Detail Normal Map
uniform vec2 uBlueTiling; //name "Blue UV Tiling" default 4,4
uniform float uBlueDetailWeight; //name "Blue Detail Weight" default 1
uniform int uFlipBlueY; //name "Flip Blue Y" bool default 0
 
uniform int uAlphaEnable; //name "Enable Alpha Map" bool default 0
USE_TEXTURE2D(tAlphaDetail); //linear name "Alpha Detail Normal Map" 
uniform vec2 uAlphaTiling; //name "Alpha UV Tiling" default 4,4
uniform float uAlphaDetailWeight; //name "Alpha Detail Weight" default 1
uniform int uFlipAlphaY; //name "Flip Alpha Y" bool default 0

vec3 sample_detail_normal(
	Texture2D detail_normal_map,
	vec2 uv,
	vec2 tiling,
	int flip_y
)
{
	vec3 detail_normal 		= textureMaterial( detail_normal_map, uv * tiling ).xyz;
	if( flip_y == 1 )
		detail_normal.y 	= 1.0 - detail_normal.y;
	detail_normal 			=  detail_normal * 2.0 - 1.0;
	return detail_normal;
};

void	SurfaceDetailNormalChannel( inout FragmentState s )
{
	SurfaceNormalMap(s);

	vec3 default_normal 	= vec3( 0.5, 0.5, 1.0 );

	vec2 uv = lerp( s.vertexTexCoord.xy, s.vertexTexCoordSecondary.xy, uCustomDetailNormalUseSecondaryUV );
	vec4 mask_map = textureMaterial( tMaskMap, uv );
	mask_map 	 *= vec4( uRedEnable * uRedDetailWeight, 
						  uGreenEnable * uGreenDetailWeight, 
						  uBlueEnable * uBlueDetailWeight, 
						  uAlphaEnable * uAlphaDetailWeight );

	vec3 red_detail_nrm 	= sample_detail_normal( tRedDetail,   s.vertexTexCoord.xy, uRedTiling,   uFlipRedY ) * mask_map.x;	
	vec3 green_detail_nrm 	= sample_detail_normal( tGreenDetail, s.vertexTexCoord.xy, uGreenTiling, uFlipGreenY ) * mask_map.y;
	vec3 blue_detail_nrm 	= sample_detail_normal( tBlueDetail,  s.vertexTexCoord.xy, uBlueTiling,  uFlipBlueY ) * mask_map.z;
	vec3 alpha_detail_nrm 	= sample_detail_normal( tAlphaDetail, s.vertexTexCoord.xy, uAlphaTiling, uFlipAlphaY ) * mask_map.w;
	
	//ortho-normalization of new tangent basis
	vec3 T = s.vertexTangent;
	vec3 B = s.vertexBitangent;
	vec3 N = s.normal;
	T -= dot(T,N)*N;
	T = normalize(T);
	B -= dot(B,N)*N + dot(B,T)*T;
	B = normalize(B);
	
	vec3 dn = red_detail_nrm + green_detail_nrm + blue_detail_nrm + alpha_detail_nrm;
	dn =	dn.x * T +
			dn.y * B +
			dn.z * N;

	s.normal = normalize( s.normal + dn );
}

#ifdef Surface
	#undef Surface
#endif
#define Surface SurfaceDetailNormalChannel