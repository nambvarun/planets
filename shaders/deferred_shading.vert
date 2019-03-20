#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;
//out vec3 rayDirection;
//uniform mat4 view;
//uniform mat4 projection;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos, 1.0);

//    vec4 reverseVec;

    /* inverse perspective projection */
//    reverseVec = vec4(gl_Vertex.xy, 0.0, 1.0);
//    reverseVec = gl_ProjectionMatrixInverse * reverseVec;

    /* inverse modelview, without translation */
//    reverseVec.w = 0.0;
//    reverseVec *= gl_ModelViewMatrixInverse;

    /* send */
//    rayDirection = vec3(reverseVec);
//    gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);

}