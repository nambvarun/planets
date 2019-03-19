#version 330 core
out vec4 fragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;

#define PI 3.14159265358979323846
#define STEP_COUNT 10
#define SIGMA 0.3

struct Light {
    vec3 position;
    vec3 direction;
    vec3 color;

    float linear;
    float quadratic;
    float radius;

    uint lightType;     // Case 0: Directional Light
                        //      1: Point Light
};

const int NR_LIGHTS = 1;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;

const float G_SCATTERING = 0.8;
const int RAY_MARCH_STEPS = 50;

// from user iq from shader toy.
float seed;
float random()  {
    return fract(sin(seed++)*43758.5453123);
}

// Ray Data Structure
struct Ray  {vec3 origin; vec3 dir;};

float Trow_Reitz_GGX(float NH, float alpha)     {
    float num = alpha * alpha;
    float denom = pow(pow(NH, 2.0) * ((alpha * alpha) - 1.0) + 1.0, 2.0);

    return num/denom;
}

float Schlick_GGX(float NV, float NL, float k)  {
    float G1 = NV/((NV)*(1.0-k) + k);
    float G2 = NL/((NL)*(1.0-k) + k);

    return G1 * G2;
}

// Mie scaterring approximated with Henyey-Greenstein phase function.
//float computeScattering(float lightDotView)     {
//    float result = 1.0f - G_SCATTERING * G_SCATTERING;
//    result /= (4.0f * PI * pow(1.0f + G_SCATTERING * G_SCATTERING - (2.0f * G_SCATTERING) * lightDotView, 1.5f));
//    return result;
//}

float BeerLambertLaw(int iterations, float extinction_coeff)    {
    return exp(-iterations * extinction_coeff);
}

vec3 fresnelSchlick(float HV, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - HV, 5.0);
}

vec3 PBR_BRDF(vec3 L, vec3 V, vec3 N, vec3 diffuse_color, vec3 light_color, vec3 F0, float alpha)
{
    vec3 lambert_component = diffuse_color * (max(dot(L, N), 0.0));

    vec3 H = normalize(L + V);
    float dotLN = max(dot(L, N), 0.);
    float dotNH = max(dot(N, H), 0.);
    float dotVN = max(dot(V, N), 0.);
    float dotHV = max(dot(H, V), 0.);

    float k = pow(alpha + 1.0, 2.0) / 8.0;

    float G = Schlick_GGX(dotVN, dotLN, k);
    float D = Trow_Reitz_GGX(dotNH, alpha);
    vec3 F = fresnelSchlick(dotHV, F0);

    float denom = max(4.0 * dotLN * dotVN, 1.0);

    vec3 cook_component = (D * G * F)/denom * (max(dot(L, N), 0.0));

    return lambert_component + cook_component;
}

void sampleEquiangular(
float u,
float maxDistance,
vec3 rayOrigin,
vec3 rayDir,
vec3 lightPos,
out float dist,
out float pdf)
{
    // get coord of closest point to light along (infinite) ray
    float delta = dot(lightPos - rayOrigin, rayDir);

    // get distance this point is from light
    float D = length(rayOrigin + delta*rayDir - lightPos);

    // get angle of endpoints
    float thetaA = atan(0.0 - delta, D);
    float thetaB = atan(maxDistance - delta, D);

    // take sample
    float t = D*tan(mix(thetaA, thetaB, u));
    dist = delta + t;
    pdf = D/((thetaB - thetaA)*(D*D + t*t));
}


void main()
{             
    // retrieve data from gbuffer
    vec3 fragPos = texture(gPosition, TexCoords).rgb;
    vec3 normal = texture(gNormal, TexCoords).rgb;
    vec3 diffuse = texture(gAlbedoSpec, TexCoords).rgb;
    float specular = texture(gAlbedoSpec, TexCoords).a;
    vec3 F0 = vec3(0.91, 0.92, 0.92);

    vec3 particleIntensity = vec3(1.0/(4.0*PI));


    // then calculate lighting as usual
    vec3 lighting  = diffuse * .1; // hard-coded ambient component
    vec3 viewDir  = normalize(viewPos - fragPos);

    for(int i = 0; i < NR_LIGHTS; ++i)
    {
        // calculate distance between light source and current fragment
        float distance = length(lights[i].position - fragPos);
        float attenuation = 1.0 / (1.0 + lights[i].linear * distance + lights[i].quadratic * distance * distance);
        if(distance < lights[i].radius)
        {
            // light direction
            vec3 lightDir = normalize(lights[i].position - fragPos);

            // attenuation
//            float attenuation = 1.0 / (1.0 + lights[i].linear * distance + lights[i].quadratic * distance * distance);
            vec3 color = lights[i].color;
            vec3 computed_light = PBR_BRDF(lightDir, viewDir, normal, lighting, color, F0, specular);

            lighting += attenuation * computed_light * color;
        }

        // Epiangular Solution
        vec3 rayOrigin = viewPos;
        vec3 rayDir = normalize(fragPos - viewPos);

        float t = length(fragPos - viewPos); // distance

        if (t > 10)
            t = 10.0;

        vec3 col = vec3(0.0);
        float offset = random();
        for (int j = 0; j < STEP_COUNT; ++j)    {
            float u = (float(j) + offset)/float(STEP_COUNT);

            float pdf;
            float x;
            sampleEquiangular(u, t, rayOrigin, rayDir, lights[i].position, x, pdf);

            pdf *= float(STEP_COUNT);

            vec3 particlePos = rayOrigin + x * rayDir;
            vec3 lightVec = lights[i].position - particlePos;
            float d = length(lightVec);

            // need to check for shadows. will do this later.
            float trans = exp(-SIGMA*(d + x));
            float geomTerm = 1.0/dot(lightVec, lightVec);
            col += SIGMA * particleIntensity * 100.0 * geomTerm * trans/pdf;
        }
        lighting += col;

    }    
    fragColor = vec4(lighting, 1.0);

//    vec3 ray = fragPos - viewPos;
//    float rayLength = length(ray);
//    vec3 rayDirection = ray / rayLength;
//
//    float stepLength = rayLength / RAY_MARCH_STEPS;
//
//    vec3 step = rayDirection * stepLength;
//    vec3 currentPosition = viewPos;
//
//    vec3 accumFog = vec3(0.0);
//
//    for (int j = 0; j < RAY_MARCH_STEPS; ++j)   {
//        vec3 lightDir = normalize(lights[i].position - currentPosition);
//
//        float rayMarchDist = length(lights[i].position - currentPosition);
//        if (rayMarchDist < lights[i].radius)
//        accumFog += BeerLambertLaw(j, stepLength) * 10 * computeScattering(dot(rayDirection, lightDir)) * lights[i].color;
//
//        currentPosition += step;
//    }
//
//    accumFog /= RAY_MARCH_STEPS;
//
//    lighting += accumFog;
}
