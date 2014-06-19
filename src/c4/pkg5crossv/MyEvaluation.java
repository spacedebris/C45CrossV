package c4.pkg5crossv;

import weka.classifiers.Evaluation;
import weka.core.Instances;


public class MyEvaluation extends Evaluation
{
	public MyEvaluation(Instances data)
	throws Exception
	{
		super(data);
	}

	public int getNumClasess()
	{
		return super.m_NumClasses;
	}

	public String getClassName(int classIndex)
	{
		return super.m_ClassNames[classIndex];
	}
}
