两种dynamic模式可选：
1，实际dynamics就是MRD,相当于已知dynamics。
运行方式，只要运行online.m即可，相当于每模拟一步就生成一步的probability

2，可以自定义dynamics，我放在了customDyn文件夹里面，只要写了Dyn就可以


两种预测方式可选：
1， online，边走边预测
2， offline 得到一堆measurements全扔给inference.m让它预测

TODO:
感觉用于MRD的参数可能需要tune，现在看这个曲线很奇怪，此外我们为了生成轨迹要不要考虑一下用ODE？ 然后在这段连续轨迹中取一些点作为measurements？