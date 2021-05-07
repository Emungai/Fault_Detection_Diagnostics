//------------------------------------------//
//  This file is a modified version of      //
//  basics.cpp, which was distributed as    //
//  part of MuJoCo,  Written by Emo Todorov //
//  Copyright (C) 2017 Roboti LLC           //
//  Modifications by Atabak Dehban          //
//------------------------------------------//

/*****************************************************************************
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/


#include <iostream>

#include "mujoco.h"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "glfw3.h"

// Libraries for sleep
#include <chrono>
#include <thread>


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
float_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum kp=25;
mjtNum kd=0;
mjtNum ctrl_q1R;
mjtNum ctrl_q2R;
mjtNum ctrl_q1L;
mjtNum ctrl_q2L;
//mjtNum ctrl;
//float qpos_d[7]={0,0,0,0.22,-0.22,0,0}; //x,z,rot_y,q1_right,q2_right,q1_left,q2_left
float qpos_d[7]={0,0,0,0,0,0,0}; //x,z,rot_y,q1_right,q2_right,q1_left,q2_left
float qvel_d[7]={0,0,0,0,0,0,0};
float ya[4];
float yd[4];
float dya[4];
float dyd[4];
float y[4];
float dy[4];

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

// control loop callback
void mycontroller(const mjModel* m, mjData* d)
{
    // printouts for debugging purposes
   /* std::cout << "number of position coordinates: " << m->nq << std::endl;
    std::cout << "number of degrees of freedom: " << m->nv << std::endl;
    std::cout << "joint position: " << d->qpos[0] << std::endl;
    std::cout << "joint velocity: " << d->qvel[0] << std::endl;
    std::cout << "Sensor output: " << d->sensordata[0] << std::endl; */

     std::cout << "y1 output: " <<y[0] << std::endl;
     std::cout << "y2 output: " <<y[1] << std::endl;
     std::cout << "y3 output: " <<y[2] << std::endl;
     std::cout << "y4 output: " <<y[3] << std::endl;

    std::cout << "dy1 output: " <<dy[0] << std::endl;
     std::cout << "dy2 output: " <<dy[1] << std::endl;
     std::cout << "dy3 output: " <<dy[2] << std::endl;
     std::cout << "dy4 output: " <<dy[3] << std::endl;

    // controller with true values, but it is cheating.
    //ctrl = 3.5*(-d->qvel[0]-10.0*d->qpos[0]);
    ya[0]=d->qpos[4];
    ya[1]=d->qpos[5];
    ya[2]=d->qpos[6];
    ya[3]=d->qpos[7];

    yd[0]=qpos_d[4];
    yd[1]=qpos_d[5];
    yd[2]=qpos_d[6];
    yd[3]=qpos_d[7];


    dya[0]=d->qvel[4];
    dya[1]=d->qvel[5];
    dya[2]=d->qvel[6];
    dya[3]=d->qvel[7];

    dyd[0]=qvel_d[4];
    dyd[1]=qvel_d[5];
    dyd[2]=qvel_d[6];
    dyd[3]=qvel_d[7];

    y[0]=yd[0]-ya[0];
    y[1]=yd[1]-ya[1];
    y[2]=yd[2]-ya[2];
    y[3]=yd[3]-ya[3];

    dy[0]=dyd[0]-dya[0];
    dy[1]=dyd[1]-dya[1];
    dy[2]=dyd[2]-dya[2];
    dy[3]=dyd[3]-dya[3];

    ctrl_q1R=kp*(y[0])-kd*dy[0];
    ctrl_q2R=kp*(y[1])-kd*dy[1];;
    ctrl_q1L=kp*(y[2])-kd*dy[2];;
    ctrl_q2L=kp*(y[3])-kd*dy[3];;
 

/*
    // controller with sensor readings
    if (previous_time == 0)
    {
        previous_time = d->time;
        return;
    }
    if (d->time - last_update > 1.0/ctrl_update_freq)
    {
        mjtNum vel = (d->sensordata[0] - position_history)/(d->time-previous_time);
        ctrl = 3.5*(-vel-10.0*d->sensordata[0]);
        last_update = d->time;
        position_history = d->sensordata[0];
        previous_time = d->time;
    }
*/
    d->ctrl[0] = ctrl_q1R;
    d->ctrl[1] = ctrl_q2R;
    d->ctrl[2] = ctrl_q1L;
    d->ctrl[3] = ctrl_q2L;
   d->ctrl[0] = 0;
    d->ctrl[1] = 0;
    d->ctrl[2] = 0;
    d->ctrl[3] = 0; 

   /* std::cout << "q1R torque effort: " << ctrl_q1R << std::endl;
    std::cout << "q2R torque effort: " << ctrl_q2R << std::endl;
    std::cout << "q1L torque effort: " << ctrl_q1L << std::endl;
    std::cout << "q2L torque effort: " << ctrl_q2L << std::endl; */
}


// main function
int main(int argc, const char** argv)
{

    // activate software
    mj_activate("/home/exo/Documents/eva/Fault_Detection_Diagnostics/MuJoCo_and_VS/mujoco200_linux/bin/mjkey.txt");
    

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 )
       m = mj_loadXML("/home/exo/Documents/eva/Fault_Detection_Diagnostics/MuJoCo_and_VS/five_link_xml_model/five_link_yukai.xml", 0, error, 1000);
        //m= mj_loadXML("/home/exo/Documents/mujoco200_linux/model/humanoid.xml", 0, error, 1000); //if no xml file is given load this file

    else
        if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
            m = mj_loadModel(argv[1], 0);
        else
            m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);


    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1244, 700, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // install control callback
    mjcb_control = mycontroller;

    // initial position
    
    //d->qpos[4] = 0.22;
    //d->qpos[5] = -0.22;

    // run main loop, target real-time simulation and 60 fps rendering
    mjtNum timezero = d->time;
    double_t update_rate = 0.01;

    // making sure the first time step updates the ctrl previous_time
    last_update = timezero-1.0/ctrl_update_freq;

    // use the first while condition if you want to simulate for a period.
//    while( !glfwWindowShouldClose(window) and d->time-timezero < 1.5)
    while( !glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
            mj_step(m, d);

        // 15 ms is a little smaller than 60 Hz.
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
       // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

          // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}
