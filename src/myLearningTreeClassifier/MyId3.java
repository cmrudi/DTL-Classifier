/*
 * Untuk Tubes 1 Machine Learning.
 */
package myLearningTreeClassifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;
import weka.core.FastVector;
import java.util.Enumeration;
import java.util.LinkedList;
import weka.core.NoSupportForMissingValuesException;

/**
 * Basic implementation of ID3 Algorithm in Machine Learning by Tom Mitchell, p56.
 * Only handle nominal values. Other values should be filtered first.
 * @author local-mreifiza
 */
public class MyId3 extends Classifier {
    /**
     * This node's attribute.
     */
    protected Attribute myAttributeLabel = null;
    /**
     * Dataset class attribute.
     */
    protected Attribute myClassLabel;
    /**
     * 
     */
    protected int[] instanceClassValueCounter;
    /**
     * This node's attribute entropy value.
     */
    protected double myEntropyValue;
    /**
     * This node's information gain value.
     */
    protected double myInfoGain;
    /**
     * Reference to root of this classifier tree model.
     */
    protected MyId3 root;
    /**
     * Reference to this node's childs.
     */
    protected MyId3[] myChilds = null;
    /**
     * Capabilities of this classifier. Only handle nominal values.
     */
    private static Capabilities myCapabilities = null;
    /**
     * Is this node a leaf?.
     */
    protected boolean isLeaf = false;
    /**
     * If this ID3 node is a leaf, then classValue is valued to something among 
     * class values, if not then it is null.
     */
    protected double classValue;
    /**
     * 
     */
    protected LinkedList<Integer> selectedIndex;
    
    @Override
    public Capabilities getCapabilities(){
        myCapabilities = super.getCapabilities();
        myCapabilities.disableAll();;
        myCapabilities.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        myCapabilities.enable(Capabilities.Capability.NOMINAL_CLASS);
        myCapabilities.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        return myCapabilities;
    }
    
    /**
     * Build basic ID3 classifier.
     * @param data
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(myCapabilities == null){
           getCapabilities();
        }
        myCapabilities.testWithFail(data);
        
        data.deleteWithMissingClass();
        myClassLabel = data.classAttribute();
        selectedIndex = new LinkedList<>();
        root = this;
        buildTree(data, this);
    }
    
   /**Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
    @Override
    public double classifyInstance(Instance instance) throws 
            NoSupportForMissingValuesException{
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                         + "please.");
        }
        double retval = Instance.missingValue(); 
        if(isLeaf){
            //System.out.println (instance);
            //System.out.println("CLASSIFIED TO " + myClassLabel.value((int)classValue));
            retval = classValue;
        }
        else{
            int index = (int) instance.value(myAttributeLabel);
            if(index >= 0 && index < myChilds.length && myChilds[index] != null){
                retval = myChilds[index].classifyInstance(instance);
            }
        }
        return retval;
    }
    
    /**
     * Print built tree model of this classifier.
     * @return String representation of this classifier tree model.
     */
    @Override
    public String toString(){
        return toStringRecursive();
    }
    
    private String toStringRecursive(){
        StringBuffer output = new StringBuffer();
        if(isLeaf){
            output.append(": "+ myClassLabel.value((int)classValue));
        }
        else{
            for (int i = 0; i < myChilds.length; i++){
                if(this != root){
                    output.append("\n|\t");
                }
                output.append(myAttributeLabel.name() + " = " + myAttributeLabel.value(i));
                if(myChilds[i] == null){
                    output.append(" : null");
                }
                else{ 
                    output.append(myChilds[i].toStringRecursive());
                }
            }
        }
        output.append("\n");
        return output.toString();
    }
    
    protected boolean isAttributeAlreadySelected(int attributeIndex){
        return selectedIndex.contains(attributeIndex);
    }
    
    protected MyId3 buildTree(Instances dataset, MyId3 rootNode){
        root = rootNode;
        //int countInstance = dataset.numInstances();
        myEntropyValue = calculateEntropy(dataset);
        //select most common value class index found in dataset instances.
        int maxIndex = 0;
        for(int i = 0; i < instanceClassValueCounter.length; i++){
            if(instanceClassValueCounter[i] > instanceClassValueCounter[maxIndex]){
                maxIndex = i;
            }
        }
        //All instance have same class value
        if(myEntropyValue == 0d){
            myInfoGain = 0d;
            isLeaf = true;
            classValue = maxIndex;
            return this;
        }
        else{    
            int countAttribute = dataset.numAttributes() - selectedIndex.size();
            //countAttribute =1, and it is assumed to be dataset classAttribute
            if(countAttribute == 1){
                myInfoGain = 0d;
                isLeaf = true;
                classValue = maxIndex;
                return this;
            }
            //otherwise, begin building child tree.
            //for each attribute, calculate its infogain.
            double[] infoGains = new double[countAttribute];
            for(int i = 0; i < countAttribute; i++){
                if(i != dataset.classIndex()){
                    infoGains[i] = calculateInfoGain(dataset, dataset.attribute(i));
                }
                else{
                    infoGains[i] = -99;
                }
            }
            //select max info gain
            int maxInfoGainIndex = 0;
            for(int i = 0; i < countAttribute; i++){
                if(!isAttributeAlreadySelected(i) && 
                    infoGains[i] > infoGains[maxInfoGainIndex]){
                    maxInfoGainIndex = i;
                }
            }
            myAttributeLabel = dataset.attribute(maxInfoGainIndex);
            myInfoGain = infoGains[maxInfoGainIndex];
            selectedIndex.add(maxInfoGainIndex);
            
            //System.out.println(myAttributeLabel);
            Instances[] dataSubsets = getSubset(dataset, myAttributeLabel);
            /*
            for (Instances dataSubset : dataSubsets) {
                dataSubset.deleteAttributeAt(maxInfoGainIndex);
            }
            */
            myChilds = new MyId3[dataSubsets.length];
            for (int i = 0 ; i < dataSubsets.length; i++){
                myChilds[i] = new MyId3();
                myChilds[i].myClassLabel = myClassLabel;
                myChilds[i].selectedIndex = new LinkedList<>();
                for(Integer el : selectedIndex){
                    myChilds[i].selectedIndex.add(el);
                }
                if(dataSubsets[i].numInstances() == 0){
                    myChilds[i].root = rootNode;
                    myChilds[i].isLeaf = true;
                    myChilds[i].myEntropyValue = 0d;
                    myChilds[i].myInfoGain = 0d;
                    myChilds[i].classValue = maxIndex;
                }
                else{
                    myChilds[i] = myChilds[i].buildTree(dataSubsets[i], rootNode);
                }
            }
            return this;
        }
    }
    
    /**
     * Calculate entropy of given dataset and automatically calculate instanceClassValueCounter 
     * (a MyId3 property).
     * Dataset is assumed to have a class index. 
     * @param dataset
     * @return 
     */
    protected double calculateEntropy(Instances dataset){
        double result = 0d;
        Enumeration enumInstances = dataset.enumerateInstances();
        int counterLength = dataset.classAttribute().numValues();
        instanceClassValueCounter = new int[counterLength];
        while(enumInstances.hasMoreElements()){
            Instance currentInstance = (Instance)enumInstances.nextElement();
            String currentClassValue = currentInstance.stringValue(currentInstance.classIndex());
            instanceClassValueCounter[dataset.classAttribute().indexOfValue(currentClassValue)]++;
        }
        int countInstance = dataset.numInstances();
        for(int i = 0; i < counterLength; i ++){
            double classProbability = 
                    (double)instanceClassValueCounter[i]/(double)countInstance;
            if(classProbability > 0d){
                result -= classProbability * Math.log(classProbability) / Math.log(2d);
            }
            else{
                result -= 0d;
            }
        }
        return result;
    }
    
    /**
     * Calculate info gain for given attribute in a dataset.
     * Dataset is assumed to have a class index, and myEntropyValue (property of this
     * ID3 tree node) already calculated. 
     * @param dataset
     * @param attr
     * @return 
     */
    protected double calculateInfoGain(Instances dataset, Attribute attr){
        double result = myEntropyValue;
        int dataSplitContainerLen = attr.numValues();
        double[] dataSplitEntropy = new double[dataSplitContainerLen];
        
        //Split dataset based on attr values.
        Instances[] dataSplitContainer = getSubset(dataset, attr);
        
        //Calculate entropy for each splitted dataset.
        for(int i = 0; i < dataSplitContainerLen; i++){
            dataSplitEntropy[i] = calculateEntropy(dataSplitContainer[i]);
        }
        
        //Calculate info gain
        int countInstance = dataset.numInstances();
        for(int i = 0 ; i < dataSplitContainerLen; i++){
            double classProbability = 
                    (double) dataSplitContainer[i].numInstances() / countInstance;
            result -= classProbability * dataSplitEntropy[i];
        }
        return result;
    }
    
    /**
     * Get all subsets of dataset instances based on possible attribute attr values.
     * All instance in the subset will all have same attr values.
     * @param dataset
     * @param attr
     * @return 
     */
    protected Instances[] getSubset(Instances dataset, Attribute attr){
        Enumeration enumInstances = dataset.enumerateInstances();
        int dataSubsetContainerLen = attr.numValues();
        Instances[] dataSubsetContainer = new Instances[dataSubsetContainerLen];
        
        //Must specify data subset attributes first in order to create subset instances from dataset
        FastVector dataSplitAttr = new FastVector();
        for(int i = 0; i < dataset.numAttributes(); i++){
            dataSplitAttr.addElement(dataset.attribute(i).copy());
        } 
        for(int i = 0; i < dataSubsetContainerLen; i++){
            dataSubsetContainer[i] = new Instances(dataset.relationName()+"_split_"+i,
                                                  dataSplitAttr,
                                                  dataset.numInstances());
            dataSubsetContainer[i].setClassIndex(dataset.classIndex());
        }
        
        //Create subset of dataset.
        while(enumInstances.hasMoreElements()){
            Instance currentInstance = (Instance)enumInstances.nextElement();
            String currentAttrValue = currentInstance.stringValue(attr);
            dataSubsetContainer[attr.indexOfValue(currentAttrValue)].add(currentInstance);
        }
        for(int i = 0; i < dataSubsetContainerLen; i++){
            dataSubsetContainer[i].compactify();
        }
        return dataSubsetContainer;
    }
}
