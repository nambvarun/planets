#version 330 core
out vec4 fragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform vec2 u_resolution;

#define PI 3.14159265358979323846
#define SIGMA_T 0.2
#define SIGMA_S 0.8
#define SAMPLES 10

struct Light {
    vec3 position;
    vec3 direction;
    vec3 color;

    float linear;
    float quadratic;
    float radius;

    float intensity;

    uint lightType;     // Case 0: Directional Light
                        //      1: Point Light
};

struct Pixel {
    vec2 coordinate;
};

struct Camera {
    vec3 eye;
    vec3 front;
    vec3 up;
    float fov;
};

struct Sphere {
    float radius;
    vec3 position;
};

uniform vec3 cameraPos;
uniform mat4 cameraRot;
uniform mat4 invCameraRot;
uniform mat4 projection;

const int NR_LIGHTS = 10;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;
uniform Camera cam;

const int NUM_SPHERES = 5;
uniform Sphere spheres[NUM_SPHERES];

const float time = 0.0;

const float G_SCATTERING = 0.8;
const int RAY_MARCH_STEPS = 50;

// from user iq from shader toy. instead of using a texture map, this gives a good enough pseudo random.
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

Pixel initPixel(in vec3 color) {
    Pixel pixel = Pixel(2.0 * gl_FragCoord.xy / u_resolution.xy - 1.0);

    float ratio = u_resolution.x / u_resolution.y;
    if (ratio > 1.0) {
        pixel.coordinate.x *= ratio;
    } else {
        pixel.coordinate.y /= ratio;
    }

    return pixel;
}

Ray initRay(in Pixel pixel, in Camera camera) {
    float focal = 1.0 / tan(radians(camera.fov) / 2.0);

    vec3 forward = normalize(camera.front);
    vec3 side = normalize(cross(forward, camera.up));
    vec3 up = normalize(cross(forward, side));

    vec3 direction = normalize(pixel.coordinate.x * side - pixel.coordinate.y * up + focal * forward);

    return Ray(camera.eye, direction);
}

void computeSphereIntersection(Ray ray, vec3 sphereOrigin, float sphereRadius, inout float rayT)  {
    float a = dot(ray.dir, ray.dir);
    float b = 2.0 * dot(ray.origin - sphereOrigin, ray.dir);
    float c = dot(ray.origin - sphereOrigin, ray.origin - sphereOrigin) - pow(sphereRadius, 2.0);
    float delta = pow(b, 2.0) - 4. * a * c;

    if (delta >= 0.0)   {
        float sqrt_delta = sqrt(delta);
        float t0 = (-b - sqrt_delta) / (2.0 * a);
        float t1 = (-b + sqrt_delta) / (2.0 * a);

        if (t0 > 0.0 && t0 < rayT)  {
            rayT = t0;
        } else if (0.0 < t1 && t1 < rayT)   {
            rayT = t1;
        }
    }
}

void intersectScene(Ray ray, inout float t)  {
    for (int i = 0; i < NUM_SPHERES; ++i)   {
        computeSphereIntersection(ray, spheres[i].position, spheres[i].radius, t);
    }
}

// Mie scaterring approximated with Henyey-Greenstein phase function.
//float computeScattering(float lightDotView)     {
//    float result = 1.0f - G_SCATTERING * G_SCATTERING;
//    result /= (4.0f * PI * pow(1.0f + G_SCATTERING * G_SCATTERING - (2.0f * G_SCATTERING) * lightDotView, 1.5f));
//    return result;
//}

//float BeerLambertLaw(int iterations, float extinction_coeff)    {
//    return exp(-iterations * extinction_coeff);
//}

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

void getSample(float zeta, float maxDistance, Ray ray, vec3 lightPos, out float final_t, out float pdf)
{

    float delta = dot(lightPos - ray.origin, ray.dir);    // projection
    float D = length(ray.origin + delta*ray.dir - lightPos);

    float theta_a = atan(maxDistance - delta, D);
    float theta_b = atan(0.0 - delta, D);

    float t = D * tan((1-zeta) * theta_a + zeta * theta_b);
    final_t = delta + t;
    pdf = D/((theta_b - theta_a)*(pow(D, 2.0) + pow(t, 2.0)));
}


void main()
{             
    // retrieve data from gbuffer
    vec3 fragPos = texture(gPosition, TexCoords).rgb;
    vec3 normal = texture(gNormal, TexCoords).rgb;
    vec3 diffuse = texture(gAlbedoSpec, TexCoords).rgb;
    float specular = texture(gAlbedoSpec, TexCoords).a;
    vec3 F0 = vec3(0.91, 0.92, 0.92);

    vec3 particleIntensity = vec3(1.0/(4.0 * PI));

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
            vec3 color = lights[i].color;
            vec3 computed_light = PBR_BRDF(lightDir, viewDir, normal, lighting, color, F0, specular);

            lighting += attenuation * computed_light * color;
        }

        // Epiangular Solution
        Pixel pixel = initPixel(diffuse);
        Ray ray = initRay(pixel, cam);
        float maxT = 100.;
        intersectScene(ray, maxT);

        vec3 volume_color = vec3(0.0);
        float offset = random();
        for (int j = 0; j < SAMPLES; ++j)    {
            float zeta = (float(j) + offset)/float(SAMPLES);    // grabbed this from shader toy.
                                                                // got weird results with just using random.
                                                                // not sure why...

            float pdf, t;
            getSample(zeta, maxT, ray, lights[i].position, t, pdf);

            vec3 particlePos = ray.origin + t * ray.dir;
            vec3 lightVec = lights[i].position - particlePos;
            float r = length(lightVec);

            // check if particle is occluded.
            float t2 = r;
            Ray ray_to_check = Ray(particlePos, normalize(lightVec));
            intersectScene(ray_to_check, t2);

            if (t2 == r)    {
                float trans = exp(-SIGMA_T * (t + r));
                volume_color += SIGMA_S * particleIntensity * lights[i].intensity * lights[i].color * 1/pow(r, 2.0) * trans/pdf/SAMPLES;
            }
        }

        lighting += pow(volume_color, vec3(1.0/2.2));        // GAMMA correction.

    }
    fragColor = vec4(lighting, 1.0);

    // OLD IMPLEMENTATION FOR RAY MARCHING
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
