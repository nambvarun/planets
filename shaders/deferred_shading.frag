#version 330 core
out vec4 fragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform vec2 u_resolution;

#define PI 3.14159265358979323846
#define STEP_COUNT 20
#define SIGMA 0.2
#define EPSILON 0.01

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

struct Pixel {
    vec2 coordinate;
    vec3 color;
};

struct Plane {
    vec3 position;
    vec3 normal;
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

const int res_x = 1280;         // TODO: set in main.cpp
const int res_y = 720;          // TODO: set in main.cpp
const int fov = 45;             // TODO: set in main.cpp

uniform vec3 cameraPos;
uniform mat4 cameraRot;
uniform mat4 invCameraRot;
uniform mat4 projection;

const int NR_LIGHTS = 2;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;
uniform Camera cam;

const int NUM_SPHERES = 1;
uniform Sphere spheres[NUM_SPHERES];

const float time = 0.0;

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

Ray getRay(in vec2 uv)  {
    Ray ray;
    vec2 iPlaneSize = 2. * tan(0.5 * fov) * vec2(res_x/res_y, 1.);
    vec2 ixy = (gl_FragCoord.xy/vec2(res_x, res_y) - 1.0) * iPlaneSize;

    ray.origin = cameraPos;
    ray.dir = (cameraRot * normalize(vec4(ixy.x, ixy.y, -1.0, 0.0))).rgb;

    return ray;
}

Pixel initPixel(in vec3 color) {
    Pixel pixel = Pixel(
        2.0 * gl_FragCoord.xy / u_resolution.xy - 1.0,                          /* coordinate */
        color                                                                   /* color */
    );

    float ratio = u_resolution.x / u_resolution.y;
    if (ratio > 1.0) {
        pixel.coordinate.x *= ratio;
    } else {
        pixel.coordinate.y /= ratio;
    }

    return pixel;
}

void sampleCamera(vec2 u, out vec3 rayOrigin, out vec3 rayDir)  {
    vec2 filmUv = (gl_FragCoord.xy + u)/u_resolution.xy;

    float tx = (2.0 * filmUv.x - 1.0)*(u_resolution.x/u_resolution.y);
    float ty = (1.0 - 2.0 * filmUv.y);
    float tz = 0.0;

    rayOrigin = cam.eye;
    rayDir = normalize(vec3(tx, ty, tz) - rayOrigin);
}

Ray initRay(in Pixel pixel, in Camera camera) {
    float focal = 1.0 / tan(radians(camera.fov) / 2.0);

    vec3 forward = normalize(camera.front);
    vec3 side = normalize(cross(forward, camera.up));
    vec3 up = normalize(cross(forward, side));

    vec3 direction = normalize(pixel.coordinate.x * side - pixel.coordinate.y * up + focal * forward);

    return Ray(
    camera.eye,                                                             /* origin */
    direction                                                               /* direction */
    );
}

float computeSphereIntersection(inout Ray ray, in Sphere sphere) {
    float a = dot(ray.dir, ray.dir);
    float b = 2.0 * dot(ray.dir, ray.origin - sphere.position);
    float c = dot(ray.origin - sphere.position, ray.origin - sphere.position) - sphere.radius * sphere.radius;
    float t = -1.0;
    float delta = b * b - 4.0 * a * c;
    if (delta >= 0.0) {
        float sqrt_delta = sqrt(delta);
        float t1 = (- b - sqrt_delta) / (2.0 * a);
        float t2 = (- b + sqrt_delta) / (2.0 * a);
        float direction = 1.0;
        if (t1 > 0.0) {
            t = t1;
        } else if (t2 > 0.0) {
            t = t2;
            direction = -1.0;
        } else {
            return t;
        }
        ray.origin = ray.origin + t * ray.dir;
        ray.dir = normalize(ray.origin - sphere.position) * direction;
    }
    return t;
}

void intersectSphere(
vec3 rayOrigin,
vec3 rayDir,
vec3 sphereCentre,
float sphereRadius,
inout float rayT)
{
    // ray: x = o + dt, sphere: (x - c).(x - c) == r^2
    // let p = o - c, solve: (dt + p).(dt + p) == r^2
    //
    // => (d.d)t^2 + 2(p.d)t + (p.p - r^2) == 0
    vec3 p = rayOrigin - sphereCentre;
    vec3 d = rayDir;
    float a = dot(d, d);
    float b = 2.0*dot(p, d);
    float c = dot(p, p) - sphereRadius*sphereRadius;
    float q = b*b - 4.0*a*c;
    if (q > 0.0) {
        float denom = 0.5/a;
        float z1 = -b*denom;
        float z2 = abs(sqrt(q)*denom);
        float t1 = z1 - z2;
        float t2 = z1 + z2;
        bool intersected = false;
        if (0.0 < t1 && t1 < rayT) {
            intersected = true;
            rayT = t1;
        } else if (0.0 < t2 && t2 < rayT) {
            intersected = true;
            rayT = t2;
        }
    }
}


float computePlaneIntersection(inout Ray ray, in Plane plane) {
    float t = -1.0;
    float d = dot(plane.normal, ray.dir);
    if (abs(d) <= EPSILON) {
        return t;
    }
    t = dot(plane.normal, plane.position - ray.origin) / d;
    ray.origin = ray.origin + t * ray.dir;
    ray.dir = -sign(d) * plane.normal;
    return t;
}

void intersectScene(Ray ray, inout float t)  {
    intersectSphere(ray.origin, ray.dir, spheres[0].position, spheres[0].radius, t);
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
        Pixel pixel = initPixel(diffuse);
        Ray ray = initRay(pixel, cam);
//        Ray ray = getRay(TexCoords);
        vec3 rayOrigin = ray.origin;
        vec3 rayDir = ray.dir;
//        sampleCamera(vec2(0.5, 0.5), rayOrigin, rayDir);
//        vec3 rayOrigin = viewPos;
//        vec3 rayDir = normalize(fragPos - viewPos);

//        float t = length(lights[i].position - viewPos); // distance
        float t = 100.;
        intersectScene(ray, t);

        //        if (t > 10)
//        float t = 100.0;

        vec3 col = vec3(0.0);
        float offset = random();
        for (int j = 0; j < STEP_COUNT; ++j)    {
            float u = (float(j) + offset)/float(STEP_COUNT);

            float pdf;
            float x;
            vec3 light_pos = lights[i].position;
            sampleEquiangular(u, t, rayOrigin, rayDir, light_pos, x, pdf);

            pdf *= float(STEP_COUNT);

            vec3 particlePos = rayOrigin + x * rayDir;
            particlePos = particlePos;
            vec3 lightVec = light_pos - particlePos;
            float d = length(lightVec);

            float t2 = d;
            Ray ray_to_check = Ray(particlePos, normalize(lightVec));
            intersectScene(ray_to_check, t2);

            // need to check for shadows. will do this later.
            if (t2 == d)    {
                float trans = exp(-SIGMA*(d + x));
                float geomTerm = 1.0/dot(lightVec, lightVec);
                col += SIGMA * particleIntensity * 100.0 * lights[i].color * geomTerm * trans/pdf;
            }
        }

        lighting += pow(col, vec3(1.0/2.2));

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
