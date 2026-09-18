#include <ros/ros.h>
#include <hand_control.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <std_msgs/String.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"

using namespace std;

/*********************************************************************************************************
int hand_id_;
std::string port_name_;
int baudrate_;
int test_flags;
serial::Serial *com_port_;
uint8_t hand_state_;
float setangle_[6];
float setforce_[6];

int Arr[12];

void arrayCallback(const std_msgs::Int32MultiArray::ConstPtr &array)
{

    int i = 0;
    // print all the remaining numbers
    for (std::vector<int>::const_iterator it = array->data.begin(); it != array->data.end(); ++it)
    {
        if (i == 0)
            printf("act_ang:");
        if (i == 6)
            printf("\nact_force:");

        Arr[i] = *it;
        printf("%d, ", Arr[i]);
        if (i == 11)
            printf("\n\n");
        i++;
    }

    return;
}
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "handcontroltopicsubscriber");
    ros::NodeHandle nh;

    //Read launch file params
    nh.getParam("inspire_hand/hand_id", hand_id_);
    nh.getParam("inspire_hand/portname", port_name_);
    nh.getParam("inspire_hand/baudrate", baudrate_);
    nh.getParam("inspire_hand/test_flags", test_flags);

    //Initialize and open serial port
    //com_port_ = new serial::Serial(port_name_, (uint32_t)baudrate_, serial::Timeout::simpleTimeout(100));

    //topic
    ros::Subscriber sub = nh.subscribe("chatter", 1000, arrayCallback);
    ros::spin();

    return (EXIT_SUCCESS);
}
****************************************************************************************************************/
int hand_id_;
std::string port_name_;
std::string port_name_glove;
int baudrate_;
int test_flags;
serial::Serial *com_port_;
serial::Serial *com_port_glove;
uint8_t hand_state_;
float setangle_[6];
float setforce_[6];
float setpos_[6];
int flag = 0;


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "handcontroltopicsubscriber");
    ros::NodeHandle nh;


    nh.getParam("inspire_hand/hand_id", hand_id_);
    nh.getParam("inspire_hand/portname", port_name_);
    nh.getParam("inspire_hand/baudrate", baudrate_);
    nh.getParam("inspire_hand/test_flags", test_flags);
    nh.getParam("inspire_hand/portname_glove",port_name_glove);
    com_port_ = new serial::Serial(port_name_, (uint32_t)baudrate_, serial::Timeout::simpleTimeout(10));
    com_port_glove = new serial::Serial(port_name_glove, 115200, serial::Timeout::simpleTimeout(10));
    
    // if(com_port_glove.isOpen())
    // {
    //     // tcflush(com_port_glove.fd(),TCIOFLUSH);
        com_port_glove->flushInput();
        com_port_glove->flushOutput();
        ros::Duration(0.1).sleep();
    // }
    std::vector<uint8_t> output;

    ros::Publisher pub = nh.advertise<std_msgs::Int32MultiArray>("chatter", 10);

    ros::Rate loop_rate(100);

    // while (ros::ok())
    // {
    //     std::vector<uint64_t> passput;
    //     if((com_port_glove.empty()))
    //     {
    //         break;
    //     }
    //     else
    //     {
    //         ros::Duration(0.01).sleep();
    //         com_port_glove->read(passput, com_port_glove->available());//缓存区数据最大长度
    //         com_port_glove->flush();
    //     }

    // }
    while (ros::ok())
    {
        
        std_msgs::Int32MultiArray array;
        array.data.clear();

        std::vector<uint8_t> input;
        int n = 0;
        while (input.empty())//一帧数据长度是54
        {
            // ros::Duration(0.005).sleep();//不加会报段错误，原因不详
            com_port_glove->read(input, 63);//com_port_glove->available()
            com_port_glove->flush();
            while(static_cast<int>(input[n]) != 170 && static_cast<int>(input[n+1]) != 85)
            {
                n++;
                if(n>63)
                {
                    n = 0;
                    break;
                }
                
            }
            ROS_INFO_STREAM("input:"
                            <<static_cast<int>(input[0])
                            <<" "
                            <<static_cast<int>(input[1])
                            <<" "
                            <<static_cast<int>(input[2])
                            <<" "
                            <<static_cast<int>(input[3])
                            <<" "
                            <<static_cast<int>(input[4])
                            <<" "
                            <<static_cast<int>(input[5])
                            <<" "
                            <<static_cast<int>(input[6])
                            <<" "
                            <<static_cast<int>(input[7])
                            <<" "
                            <<static_cast<int>(input[8])
                            <<" "
                            <<static_cast<int>(input[9])
                            <<" "
                            <<static_cast<int>(input[10])
                            <<" "
                            <<static_cast<int>(input[11])
                            <<" "
                            <<static_cast<int>(input[12])
                            <<" "
                            <<static_cast<int>(input[13])
                            <<" "
                            <<static_cast<int>(input[14])
                            <<" "
                            <<static_cast<int>(input[15])<<"************************");           
        }
            // n = 0;
            int temp[6] = {0};
            for (int j = 0; j < 6; j++)
            {
                temp[j] = (((input[n + 4 + j * 2] << 8) & 0xff00) + (input[n + 3 + j * 2] & 0x00ff));
            }
            ROS_INFO_STREAM("hand: set pos: "
                            << temp[0]
                            << " "
                            << temp[1]
                            << " "
                            << temp[2]
                            << " "
                            << temp[3]
                            << " "
                            << temp[4]
                            << " "
                            << temp[5]);
            setpos_[0] = float((int(temp[0]) - 1020) * 1000/782);//602为握拳时set pos值，871为张开手掌和握拳时set pos的差值
            setpos_[1] = float((temp[1] - 1145) * 1000/707);//同上
            setpos_[2] = float((temp[2] - 1211) * 1000/648);//同上
            setpos_[3] = float((temp[3] - 1046) * 1000/702);//同上
            setpos_[4] = float((temp[4] - 948) * 1000/400);//同上
            setpos_[5] = float((temp[5] - 911) * 1000/394);//同上
            for(int i = 0;i<6;i++)
            {
                if(setpos_[i]<0)
                    setpos_[i] = 0;
                if (setpos_[i] > 1000)
                    setpos_[i] = 1000;
            }
            ROS_INFO_STREAM("hand: ACT pos: "
                            << setpos_[0]
                            << " "
                            << setpos_[1]
                            << " "
                            << setpos_[2]
                            << " "
                            << setpos_[3]
                            << " "
                            << setpos_[4]
                            << " "
                            << setpos_[5]);
            for (int i = 0; i <6; i++)
            {
                array.data.push_back(setpos_[i]);
            }
        if(static_cast<int>(input[n]) == 170 && static_cast<int>(input[n+1]) == 85)
            pub.publish(array);
        input.clear();

        loop_rate.sleep();
        

        
    }

    return(EXIT_SUCCESS);
}
