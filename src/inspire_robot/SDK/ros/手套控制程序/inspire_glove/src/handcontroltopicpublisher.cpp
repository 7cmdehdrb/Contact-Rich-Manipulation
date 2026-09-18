#include <ros/ros.h>
#include <hand_control.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <std_msgs/String.h>
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"

using namespace std;

/*************************************************************************************
int hand_id_;
std::string port_name_;
int baudrate_;
int test_flags_;
int Serial_flags_;
serial::Serial *com_port_;
int hand_state_;
float curangle_[6];
float curforce_[6];

void getANGLE_ACT1(serial::Serial *port)
{
    std::vector<uint8_t> output;
    //message from master to module
    output.push_back(0xEB);
    output.push_back(0x90);
    //module id
    output.push_back(hand_id_);
    //Data Length
    output.push_back(0x01);
    //Command get state
    output.push_back(0xD7);
    //Checksum calculation
    unsigned int check_num = 0;
    int len = output[3] + 5;
    for (int i = 2; i < len - 1; i++)
        check_num = check_num + output[i];
    //Add checksum to the output buffer
    output.push_back(check_num & 0xff);
    //Send message to the module
    port->write(output);

    ros::Duration(0.015).sleep();

    std::string s1;
    for (int i = 0; i < output.size(); ++i)
    {
        char str[16];
        sprintf(str, "%02X", output[i]);
        s1 = s1 + str + " ";
    }
    if (test_flags_ == 1)
        ROS_INFO_STREAM("Write: " << s1);

    //Read response
    std::vector<uint8_t> input;
    while (input.empty())
    {
        port->read(input, (size_t)64);
    }

    std::string s2;
    for (int i = 0; i < input.size(); ++i)
    {
        char str[16];
        sprintf(str, "%02X", input[i]);
        s2 = s2 + str + " ";
    }
    if (test_flags_ == 1)
        ROS_INFO_STREAM("Read: " << s2);

    float temp[10] = {0.0};
    for (int j = 0; j < 6; j++)
        temp[j] = ((input[6 + j * 2] << 8) & 0xff00) + input[5 + j * 2];
    curangle_[0] = temp[0];
    curangle_[1] = temp[1];
    curangle_[2] = temp[2];
    curangle_[3] = temp[3];
    curangle_[4] = temp[4];
    curangle_[5] = temp[5];
}

void getFORCE_ACT1(serial::Serial *port)
{
    std::vector<uint8_t> output;
    //message from master to module
    output.push_back(0xEB);
    output.push_back(0x90);
    //module id
    output.push_back(hand_id_);
    //Data Length
    output.push_back(0x04);
    //Command get state
    output.push_back(0x11);
    output.push_back(0x2E);
    output.push_back(0x06);
    output.push_back(0x0C);
    //Checksum calculation
    unsigned int check_num = 0;
    int len = output[3] + 5;
    for (int i = 2; i < len - 1; i++)
        check_num = check_num + output[i];
    //Add checksum to the output buffer
    output.push_back(check_num & 0xff);
    //Send message to the module
    port->write(output);

    ros::Duration(0.015).sleep();

    std::string s1;
    for (int i = 0; i < output.size(); ++i)
    {
        char str[16];
        sprintf(str, "%02X", output[i]);
        s1 = s1 + str + " ";
    }
    if (test_flags_ == 1)
        ROS_INFO_STREAM("Write: " << s1);

    //Read response
    std::vector<uint8_t> input;
    while (input.empty())
    {
        port->read(input, (size_t)64);
    }
    std::string s2;
    for (int i = 0; i < input.size(); ++i)
    {
        char str[16];
        sprintf(str, "%02X", input[i]);
        s2 = s2 + str + " ";
    }
    if (test_flags_ == 1)
        ROS_INFO_STREAM("Read: " << s2);

    int temp[10] = {0};
    for (int j = 0; j < 6; j++)
    {
        temp[j] = ((input[8 + j * 2] << 8) & 0xff00) + input[7 + j * 2];
        if (temp[j] > 32768)
            temp[j] = temp[j] - 65536;
    }

    curforce_[0] = float(temp[0]);
    curforce_[1] = float(temp[1]);
    curforce_[2] = float(temp[2]);
    curforce_[3] = float(temp[3]);
    curforce_[4] = float(temp[4]);
    curforce_[5] = float(temp[5]);
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "handcontroltopicpublisher");
    ros::NodeHandle nh;

    //topic
    ros::Publisher chatter_pub = nh.advertise<std_msgs::Int32MultiArray>("chatter", 1000);

    ros::Rate loop_rate(10);

    //Read launch file params
    nh.getParam("inspire_hand/hand_id", hand_id_);
    nh.getParam("inspire_hand/portname", port_name_);
    nh.getParam("inspire_hand/baudrate", baudrate_);
    nh.getParam("inspire_hand/test_flags", test_flags_);
    
    //Initialize and open serial port
    com_port_ = new serial::Serial(port_name_, (uint32_t)baudrate_, serial::Timeout::simpleTimeout(100));

    while (ros::ok())
    {
        //get the param Serial_flags
        nh.getParam("inspire_hand/Serial_flags", Serial_flags_);
        if(Serial_flags_ == 0)
        {
            ROS_INFO_STREAM("close the Serial");
        }
        else if (Serial_flags_ == 1)
        {
            std_msgs::Int32MultiArray array;

            //Clear array
            array.data.clear();
            getANGLE_ACT1(com_port_);
            getFORCE_ACT1(com_port_);

            for (int i = 0; i < 6; i++)
            {
                //assign array a random number between 0 and 255.
                array.data.push_back(curangle_[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                //assign array a random number between 0 and 255.
                array.data.push_back(curforce_[i]);
            }
            //Publish array
            chatter_pub.publish(array);
            loop_rate.sleep();
        }
    }

    return (EXIT_SUCCESS);
}
*******************************************************************************************************/
int hand_id_;
std::string port_name_;
std::string port_name_glove;
int baudrate_;
int test_flags;
serial::Serial *com_port_;
serial::Serial *com_port_glove;
uint8_t hand_state_;
// void openData(serial::Serial *port)
// {
//     std::vector<uint8_t> output;
//     output.push_back(0xEB);
//     output.push_back(0x90);
//     output.push_back(0xAA);
//     output.push_back(0xFF);
//     output.push_back(0xAA);
//     ROS_INFO(output);
//     port->write(output);
// }
void setPOS1(serial::Serial *port, int pos0, int pos1, int pos2, int pos3, int pos4, int pos5) 		
{
    std::vector<uint8_t> output;
    //message from master to module
    output.push_back(0xEB);
    output.push_back(0x90);
    //module id
    output.push_back(0xFF);
    //Data Length
    output.push_back(0x0F);
    //Command get state
    output.push_back(0x12);
    output.push_back(0xCE);
    output.push_back(0x05);

    unsigned int temp_int1, temp_int2, temp_int3, temp_int4, temp_int5, temp_int6;
    temp_int1 = (unsigned int)pos0;
    temp_int2 = (unsigned int)pos1;
    temp_int3 = (unsigned int)pos2;
    temp_int4 = (unsigned int)pos3;
    temp_int5 = (unsigned int)pos4;
    temp_int6 = (unsigned int)pos5;

    output.push_back(temp_int1 & 0xff);
    output.push_back((temp_int1 >> 8) & 0xff);
    
    output.push_back(temp_int2 & 0xff);
    output.push_back((temp_int2 >> 8) & 0xff);
    
    output.push_back(temp_int3 & 0xff);
    output.push_back((temp_int3 >> 8) & 0xff);
    
    output.push_back(temp_int4 & 0xff);
    output.push_back((temp_int4 >> 8) & 0xff);
    
    output.push_back(temp_int5 & 0xff);
    output.push_back((temp_int5 >> 8) & 0xff);
    
    output.push_back(temp_int6 & 0xff);
    output.push_back((temp_int6 >> 8) & 0xff);

    unsigned int check_num = 0;
    int len = output[3] + 5;

    for (int i = 2; i < len - 1; i++)
        check_num = check_num + output[i];

    output.push_back(check_num & 0xff);

    port->write(output);

    ros::Duration(0.015).sleep();

    std::string s1;

    for (int i = 0; i<output.size(); ++i)
    {
        char str[16];
        sprintf(str, "%02X", output[i]);
        s1 = s1 + str + " ";
    }
    if (test_flags == 1)
        ROS_INFO_STREAM("Write: " << s1);

    std::vector<uint8_t> input;

    while (input.empty())
    {
        port->read(input, (size_t)64);
    }

    std::string s2;
    for (int i = 0; i<input.size(); ++i)
    {
        char str[16];
        sprintf(str, "%02X", input[i]);
        s2 = s2 + str + " ";
    }
    if (test_flags == 1)
        ROS_INFO_STREAM("Read: " << s2);

}
int Arr[12];

void arrayCallback1(const std_msgs::Int32MultiArray::ConstPtr& array)
{
    int i = 0;
    for(std::vector<int>::const_iterator it = array->data.begin(); it != array->data.end(); ++it)
    {

        Arr[i] = *it;
        printf("%d, ", Arr[i]);
        i++;
    }
    // if(array.size == 5)
    // {
    //     openData(com_port_glove);
    // }
    // else
    // {
        setPOS1(com_port_,Arr[0],Arr[1],Arr[2],Arr[3],Arr[4],Arr[5]);
    // }

    return;
}
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "handcontroltopicpublisher");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("chatter", 10, arrayCallback1);

    ros::Rate loop_rate(1000);

    nh.getParam("inspire_hand/hand_id", hand_id_);
    nh.getParam("inspire_hand/portname", port_name_);
    nh.getParam("inspire_hand/baudrate", baudrate_);
    
    nh.getParam("inspire_hand/portname_glove",port_name_glove);
    
    com_port_ = new serial::Serial(port_name_, (uint32_t)baudrate_, serial::Timeout::simpleTimeout(10));
    com_port_glove = new serial::Serial(port_name_glove, 115200, serial::Timeout::simpleTimeout(10));
    
    ros::spin();

    return(EXIT_SUCCESS);
}
