#include <ros/ros.h>
#include <serial/serial.h>
// #include <inspire_hand/set_pos.h>

// #include <cstdlib>
// #include "hand_control.h"
// //#include <vector>
// #include <fstream>
// #include <iostream>
// #include <ctime>

using namespace std;

std::string port_name_glove;
int main(int argc, char** argv)
{
    ros::init(argc, argv, "glove_receive");
    ros::NodeHandle nh;

    nh.getParam("inspire_hand/portname_glove",port_name_glove);
    
    serial::Serial com_port_glove(port_name_glove,115200,serial::Timeout::simpleTimeout(100));

    if (com_port_glove.isOpen())// && com_port_glove->isOpen()
    {
        ROS_INFO_STREAM("glove: Serial port " << port_name_glove << " openned");
    }
    if(!com_port_glove.isOpen())
    {
        ROS_ERROR("fail to openUSB1");
        return 1;
    }
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        // char c = com_port_glove.getRI();
        // if(c != '\0')
        // {
        //     ROS_INFO("%c",c);
        // }
        loop_rate.sleep();
    }
    

    // serial::Timeout to = serial::Timeout::simpleTimeout(100);
    // //设置要打开的串口名称
    // com_port_glove.setPort("/dev/ttyUSB1");//通常是ttyUSB*,这里我因为串口被我绑定了，所以名字有点不一样
    // //设置串口通信的波特率
    // com_port_glove.setBaudrate(115200);
    // //串口设置timeout
    // com_port_glove.setTimeout(to);
 
    // try
    // {
    //     //打开串口
    //     com_port_glove.open();
    // }
    // catch(serial::IOException& e)
    // {
    //     ROS_ERROR_STREAM("Unable to open port.");
    //     return -1;
    // }
    
    // //判断串口是否打开成功
    // if(sp.isOpen())
    // {
    //     ROS_INFO_STREAM("/dev/ttyUSB1 is opened.");
    // }
    // else
    // {
    //     return -1;
    // }
    
    // ros::Rate loop_rate(1);
    // uint8_t buffer[30];

    // while(ros::ok())
    // {
    //     //获取缓冲区内的字节数
    //     size_t n = sp.available();
    //     // sp.write(all_data, 7);//发送内容，第一个参数是要发送的数据，也就是上面的写的那7个16进制的数，第二个参数就是数据的长度
    //     if(n!=0)
    //     {
    //         //读出数据
    //         n = sp.read(buffer, n);//读取串口返回的数据
            
    //         for(int i=0; i<n; i++)
    //         {
    //             //16进制的方式打印到屏幕
    //             std::cout << std::hex << (buffer[i] & 0xff) << " ";
    //         }
    //         std::cout << std::endl;
    //         //把数据发送回去

    //     }
    //     int temp[6] = {0};
    //     for (int j = 0; j < 6; j++)
    //         temp[j] = ((buffer[4 + j * 2] << 8) & 0xff00) + buffer[3 + j * 2];

    //     ROS_INFO_STREAM("glove: ADC: "
    //                     << temp[0]
    //                     << " "
    //                     << temp[1]
    //                     << " "
    //                     << temp[2]
    //                     << " "
    //                     << temp[3]
    //                     << " "
    //                     << temp[4]
    //                     << " "
    //                     << temp[5]);
        
    //     serial::Serial sp_hand;
    //     serial::Timeout to = serial::Timeout::simpleTimeout(100);
    //     //设置要打开的串口名称
    //     sp_hand.setPort("/dev/ttyUSB0");
    //     //设置串口通信的波特率
    //     sp_hand.setBaudrate(115200);
    //     //串口设置timeout
    //     sp_hand.setTimeout(to);

    //     std::vector<uint8_t> output;
    //     //message from master to module
    //     output.push_back(0xEB);
    //     output.push_back(0x90);
    //     //module id
    //     output.push_back(0xFF);
    //     //Data Length
    //     output.push_back(0x0F);
    //     //Command get state
    //     output.push_back(0x12);
    //     output.push_back(0xC2);
    //     output.push_back(0x05);

    //     unsigned int temp_int1, temp_int2, temp_int3, temp_int4, temp_int5, temp_int6;
    //     temp_int1 = (unsigned int)temp[0];
    //     temp_int2 = (unsigned int)temp[1];
    //     temp_int3 = (unsigned int)temp[2];
    //     temp_int4 = (unsigned int)temp[3];
    //     temp_int5 = (unsigned int)temp[4];
    //     temp_int6 = (unsigned int)temp[5];

    //     output.push_back(temp_int1 & 0xff);
    //     output.push_back((temp_int1 >> 8) & 0xff);
    //     output.push_back(temp_int2 & 0xff);
    //     output.push_back((temp_int2 >> 8) & 0xff);
    //     output.push_back(temp_int3 & 0xff);
    //     output.push_back((temp_int3 >> 8) & 0xff);
    //     output.push_back(temp_int4 & 0xff);
    //     output.push_back((temp_int4 >> 8) & 0xff);
    //     output.push_back(temp_int5 & 0xff);
    //     output.push_back((temp_int5 >> 8) & 0xff);
    //     output.push_back(temp_int6 & 0xff);
    //     output.push_back((temp_int6 >> 8) & 0xff);
    //     //Checksum calculation
    //     unsigned int check_num = 0;
    //     int len = output[3] + 5;

    //     for (int i = 2; i < len - 1; i++)
    //         check_num = check_num + output[i];
    //     //Add checksum to the output buffer
    //     output.push_back(check_num & 0xff);
    //     //Send message to the module
    //     sp_hand.write(output);
    //     ros::Duration(0.015).sleep();
    //     loop_rate.sleep();
    // }
    
    // //关闭串口
    // sp.close();
 
    return 0;
}