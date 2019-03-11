#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;

struct Light {
    vec3 Position;
    vec3 Color;
    
    float Linear;
    float Quadratic;
    float Radius;
};

const int NR_LIGHTS = 4;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;

float Trow_Reitz_GGX(float NH, float alpha)   {
    float num = alpha * alpha;
    float denom = pow(pow(NH, 2.0) * ((alpha * alpha) - 1.0) + 1.0, 2.0);

    return num/denom;
}

float Schlick_GGX(float NV, float NL, float k)  {
    float G1 = NV/((NV)*(1.0-k) + k);
    float G2 = NL/((NL)*(1.0-k) + k);

    return G1 * G2;
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

void main()
{             
    // retrieve data from gbuffer
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    vec3 Normal = texture(gNormal, TexCoords).rgb;
    vec3 Diffuse = texture(gAlbedoSpec, TexCoords).rgb;
    float Specular = texture(gAlbedoSpec, TexCoords).a;
    vec3 F0 = vec3(0.91, 0.92, 0.92);

    // then calculate lighting as usual
    vec3 lighting  = Diffuse * 0.1; // hard-coded ambient component
    vec3 viewDir  = normalize(viewPos - FragPos);
    for(int i = 0; i < NR_LIGHTS; ++i)
    {
        // calculate distance between light source and current fragment
        float distance = length(lights[i].Position - FragPos);
        if(distance < lights[i].Radius)
        {
            // light direction
            vec3 lightDir = normalize(lights[i].Position - FragPos);

            // attenuation
            float attenuation = 1.0 / (1.0 + lights[i].Linear * distance + lights[i].Quadratic * distance * distance);
            vec3 color = lights[i].Color;
            vec3 computed_light = PBR_BRDF(lightDir, viewDir, Normal, lighting, color, F0, Specular);

//            diffuse *= attenuation;
//            specular *= attenuation;
//            lighting += diffuse + specular;
            lighting += attenuation * computed_light * color;
        }
    }    
    FragColor = vec4(lighting, 1.0);
}
