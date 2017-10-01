package myLearningTreeClassifier;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class MyC45 extends Classifier {
	
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
    protected MyC45 root;
    /**
     * Reference to this node's childs.
     */
    protected MyC45[] myChilds = null;
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
    
    protected List<MyC45> listOfPrunedCandidate = null;
    protected Evaluation evaluator;
    protected double lastAccuracy;
    
    protected double gainRatio;
    // manually update
    protected boolean isUseGainRatio = false;
    
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
        prune(data);
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
    
    protected MyC45 buildTree(Instances dataset, MyC45 rootNode){
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
            double[] gainRatio = new double[countAttribute];
            for(int i = 0; i < countAttribute; i++){
                if(i != dataset.classIndex()){
                    infoGains[i] = calculateInfoGain(dataset, dataset.attribute(i));
                    gainRatio[i] = calculateGainRatio(dataset, i, infoGains[i]);
                }
                else{
                    infoGains[i] = -99;
                    gainRatio[i] = -99;
                }
            }
            //select max info gain
            int maxInfoGainIndex = 0;
            int maxGainRatioIndex = 0;
            for(int i = 0; i < countAttribute; i++){
                if(!isAttributeAlreadySelected(i) && 
                    infoGains[i] > infoGains[maxInfoGainIndex]){
                    maxInfoGainIndex = i;
                }
                if(!isAttributeAlreadySelected(i) && 
                        gainRatio[i] > gainRatio[maxGainRatioIndex]){
                        maxGainRatioIndex = i;
                }
                
            }
            
            if (isUseGainRatio) {
            	myAttributeLabel = dataset.attribute(maxGainRatioIndex);
                myInfoGain = infoGains[maxGainRatioIndex];
                selectedIndex.add(maxGainRatioIndex);
            } else {
            	myAttributeLabel = dataset.attribute(maxInfoGainIndex);
                myInfoGain = infoGains[maxInfoGainIndex];
                selectedIndex.add(maxInfoGainIndex);
            }
            
            
            //System.out.println(myAttributeLabel);
            Instances[] dataSubsets = getSubset(dataset, myAttributeLabel);
            /*
            for (Instances dataSubset : dataSubsets) {
                dataSubset.deleteAttributeAt(maxInfoGainIndex);
            }
            */
            myChilds = new MyC45[dataSubsets.length];
            for (int i = 0 ; i < dataSubsets.length; i++){
                myChilds[i] = new MyC45();
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
     * (a MyC45 property).
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
    
    protected double calculateGainRatio(Instances dataset, int attributeIdx, double infoGain) {
    	double[] test = dataset.attributeToDoubleArray(attributeIdx);
    	TreeMap<Double, Integer>tm = new TreeMap<Double,Integer>();
    	for (int i = 0; i < test.length; i++) {
    		if (tm.containsKey(test[i])) {
    			tm.put(test[i], tm.get(test[i]) + 1);
    		} else {
    			tm.put(test[i], 1);
    		}
    	}
    	double informationSplit = 0;
    	int numInstances = test.length;
    	for (int i = 0; i < tm.size(); i++) {
    		informationSplit += (-1)*(tm.get(i)/numInstances)
    				*((Math.log(tm.get(i)/numInstances))/(Math.log(2d)));
    	}
    	
    	return infoGain/informationSplit;
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
    
    protected void prune(Instances instances) {
    	try  {
	    	listOfPrunedCandidate = new ArrayList<>();
	    	this.evaluator = new Evaluation(instances);
	    	this.evaluator.evaluateModel(this, instances);
	    	this.lastAccuracy = 0.0;
	    	recursive_prune(this, instances, this);
    	}
	    catch (Exception e) {
	    	System.out.println(e);
	    }
    }
    
    protected void recursive_prune(MyC45 myC45Root, Instances instances,  MyC45 myC45) throws Exception {
    	for (int i = 0; i < myC45.myChilds.length; i++) {
    		if (!myC45.myChilds[i].isLeaf) {
	    		if (myC45.myChilds[i].isChildContainNonLeafNode()) {
	    			myC45.myChilds[i].isLeaf = true;
	    			myC45Root.evaluator.evaluateModel(myC45Root, instances);
	    			if (!(myC45.evaluator.pctCorrect() > myC45.lastAccuracy)) {
	    				myC45.myChilds[i].isLeaf = false;
	    			}
	    			
	    		} else {
	    			recursive_prune(myC45Root, instances, myC45.myChilds[i]);
	    		}
    		}
    	}
    }
    
    protected boolean isChildContainNonLeafNode() {
    	boolean pruneable = true;
    	for (int i = 0; i < this.myChilds.length; i++) {
    		if (!myChilds[i].isLeaf) {
    			pruneable = false;
    		}
    	}
    	return pruneable;
    }
    
    public static void printTree(MyC45 myC45, int indentation) throws Exception {
    	for (int i = 0; i < indentation; i++) {
    		System.out.print("-");
    	}
    	
    	System.out.println(myC45.myAttributeLabel.name() + "  numChild=" + myC45.myChilds.length);
    	
    	for (int i = 0; i < myC45.myChilds.length; i++) {
    		if (!myC45.myChilds[i].isLeaf) {
    			printTree(myC45.myChilds[i], indentation + 2);
    		} else {
    			for (int j = 0; j < indentation+2; j++) {
    	    		System.out.print("-");
    	    	}
    			System.out.println(myC45.myChilds[i].myClassLabel.name());
    		}
    	}
    }
    
    public static void main(String args[]) {
    
    	try {
    		
	    	Instances data = DataSource.read("/home/asus/Semester7/ML/weka-3-6-14/data/"
	                + "weather.nominal.arff");
	        if(data.classIndex() == -1)
	            data.setClassIndex(data.numAttributes() - 1);
	        
	        //mengapply filter. Numeric To Nominal
	        String rawOptions = "-R first-4";
	        String options[] = Utils.splitOptions(rawOptions);
	        NumericToNominal toNominalFilter = new NumericToNominal();
	        toNominalFilter.setOptions(options);
	        toNominalFilter.setInputFormat(data);
	        Instances resultFilter = Filter.useFilter(data, toNominalFilter);
	        Instances resultFiltercopy = new Instances(resultFilter);
	        //System.out.println(data.toSummaryString());
	        
	        //Melakukan pembelajaran dataset dengan skema 10-fold cross validation
	        Evaluation evaluator = new Evaluation(resultFilter);
	        MyC45 c45 = new MyC45();
	        c45.buildClassifier(resultFilter);
	        //c45.printTree(c45,0);
	        
	        Evaluation evaluat = new Evaluation(resultFilter);
	    	evaluat.evaluateModel(c45, resultFilter);
	    	//System.out.println(evaluat.toSummaryString());
	    	double[] test = data.attributeToDoubleArray(1);
	    	for (int i = 0; i < test.length; i++) {
	    		System.out.println(test[i]);
	    	}
	    	System.out.println(data.attributeToDoubleArray(1));
			
    	} catch (Exception e) {
    		System.out.println(e);
    	}
    	
	}
    
    protected Instances removeMissingValues(Instances data) {
    	for (int i = data.numInstances() - 1; i >= 0; i--) {
    	    Instance inst = data.instance(i);
    	    if (inst.classIsMissing()) {
    	        data.delete(i);
    	    }
    	    i--;
    	}
    	return data;
    }
    
    
}
