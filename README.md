# Cooperative Carrying Control for Mobile Robots in Indoor Scenario
In recent years, there has been a growing interest in designing multi­robot systems to provide cost­effective, fault­tolerant and reliable solutions to a variety of automated applications. In particular, from an industrial perspective, cooperative carrying techniques based on Reinforcement Learning (RL) gained a strong interest. Compared to a single robot system, this approach improves the system’s robustness and manipulation dexterity in the transportation of large objects. However, in the current state of the art, the environments’ dynamism and re­training procedure represent a considerable limitation for most of the existing cooperative carrying RL­based solutions. In this thesis, we employ the Value Propagation Network (VPN) algorithm for cooperative multi­-robot transport scenarios. We extend and test the ∆Q cooperation metric to V­value­based agents, and we investigate path generation algorithms and trajectory tracking controllers for differential drive robots. Moreover, we explore localization algorithms in order to take advantage of range sensors and mitigate the drift errors of wheel odometry, and we conduct experiments to derive key performance indicators of range sensors’ precision. Lastly, we perform realistic industrial indoor simulations using Robot Operating System (ROS) and Gazebo 3D visualization tool, including physical objects and 6Gcommunication constraints. Our results showed that the proposed VPN­based algorithm outperforms the current state-of-­the-­art since the trajectory planning and dynamic obstacle avoidance are performed in real-­time, without re­training the model, and under constant 6G network coverage.

# Features
- Designed an adapted version of the Value Propagation Networks algorithm for multi-robot motion planning in cooperative carrying tasks.
- Demonstrated the effectiveness of RL algorithms over traditional approaches in multi-agent scenarios and achieved superior adaptability in dynamic environments without requiring re-training procedures.
- Conducted proof-of-concept implementation using modern RL software frameworks and rigorous evaluation with traditional and novel metrics against state-of-the-art approaches.
- Built a comprehensive 3D simulation platform that accurately reflects real-world robot operations, including path generation, trajectory tracking control systems and localization algorithm (AMCL).
- Explored the integration of 6G signal connectivity for enhanced route planning, and conducted preliminary tests on 6G sensing devices for robot localization.