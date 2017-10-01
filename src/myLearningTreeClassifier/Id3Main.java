/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myLearningTreeClassifier;

import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author local-mreifiza
 */
public class Id3Main {
    public static void main(String[] args){
        try {
            //membaca dataset yang diberikan, diberikan dari mana?
            Instances data = DataSource.read("/home/asus/Semester7/ML/weka-3-6-14/data/"
                    + "iris.arff");
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
            Classifier wekaID3   = new Id3();
            Classifier myID3     = new MyId3();
            wekaID3.buildClassifier(resultFilter);
            myID3.buildClassifier(resultFilter);
            
            //System.out.println("Weka-ID3 tree model");
            //System.out.println(wekaID3);
            //System.out.println();
            //System.out.println("My-ID3 tree model");
            //System.out.println(myID3);
            
            System.out.println(resultFilter.toSummaryString());
            
            evaluator.crossValidateModel(wekaID3, resultFiltercopy, 7, new Random(103057));
            System.out.println(evaluator.toSummaryString("Weka-Id3 10-fold result", false));
            
            evaluator = new Evaluation(resultFilter);
            evaluator.crossValidateModel(myID3, resultFiltercopy, 7, new Random(103057));
            System.out.println(evaluator.toSummaryString("my-Id3 10-fold result", false));
            //evaluator.evaluateModel(myID3, resultFiltercopy);
            //System.out.println(evaluator.toSummaryString("my-Id3 full training result", false));
            
            /*
            //Melakukan pembelajaran dataset dengan skema full-training
            Evaluation evaluator2 = new Evaluation(resultFilter);
            Classifier cls = new Id3();
            cls.buildClassifier(resultFilter);
            evaluator2.evaluateModel(cls, resultFiltercopy);
            System.out.println(evaluator2.toSummaryString("Weka-Id3 Full training result", false));
            
            /*
            //Menyimpan model pembelajaran skema 10-fold dan Menyimpan model 
            //pembelajaran skema full-training
            SerializationHelper.write("D:\\Tugas\\4.5\\Artific. Intel\\TUCIL2\\"
                    + "wekawekaweka\\output\\10fold.model", nb);
             SerializationHelper.write("D:\\Tugas\\4.5\\Artific. Intel\\"
                     + "TUCIL2\\wekawekaweka\\output\\fulltraining.model", cls);

            //lalu dibaca lagi
            Classifier read10fold = (Classifier) SerializationHelper.read(
                    "D:\\Tugas\\4.5\\Artific. Intel\\TUCIL2\\wekawekaweka\\"
                            + "output\\10fold.model");
            Classifier readfull =  (Classifier) SerializationHelper.read(
                    "D:\\Tugas\\4.5\\Artific. Intel\\TUCIL2\\wekawekaweka\\"
                            + "output\\fulltraining.model");
            //Membuat instance baru sesuai masukan dari pengguna untuk setiap nilai
            //atribut
            Scanner readInput = new Scanner(System.in);
            double[] attribs = new double[data.numAttributes()];
            System.out.println("Ada " + (data.numAttributes()-1) + " atribut yang "
                    + "nilainya mesti diisi (format: float) hehe");
            for(int i = 0 ; i < attribs.length-1; i++){
                System.out.println("masukin nilai atribut instans ke-" + 
                        (i+1) + " hehe: ");
                attribs[i] = readInput.nextDouble();
            }
            Random randomz = new Random(103901);
            Instance something = new DenseInstance(randomz.nextDouble(), attribs);
            //buat instances baru
            Attribute sepallength = new Attribute("sepallength");	
            Attribute sepalwidth = new Attribute("sepalwidth");
            Attribute petallength = new Attribute("petallength");	
            Attribute petalwidth = new Attribute("petalwidth");
            ArrayList<String> clabel = new ArrayList<>();
            clabel.add("Iris-setosa");
            clabel.add("Iris-versicolor");
            clabel.add("Iris-virginica");
            Attribute classification = new Attribute("class", clabel);
            ArrayList<Attribute> attinfo = new ArrayList<>();
            attinfo.add(sepallength);
            attinfo.add(sepalwidth);
            attinfo.add(petallength);
            attinfo.add(petalwidth);
            attinfo.add(classification);
            Instances unlabeled = new Instances("userinput", attinfo, 0);
            if(unlabeled.classIndex() == -1)
                unlabeled.setClassIndex(data.numAttributes() - 1);
            unlabeled.add(something);
            //Melakukan klasifikasi dengan memanfaatkan model/hipotesis dan 
            //instance sesuai masukan pengguna pada g.
            double fulltrainresult = readfull.classifyInstance(unlabeled.instance(0));
            double tenfoldresult = read10fold.classifyInstance(unlabeled.instance(0));
            Instances wannatenfold = new Instances(unlabeled);
            Instances wannafulltrain = new Instances(unlabeled);
            wannatenfold.instance(0).setClassValue(tenfoldresult);
            wannafulltrain.instance(0).setClassValue(fulltrainresult);
            System.out.println("Berdasarkan skema 10-fold, hasil klasifikasi "
                    + "adalah: " + wannatenfold.instance(0).stringValue(classification));
            System.out.println("Berdasarkan skema full training set, "
                    + "hasil klasifikasi adalah: " + wannafulltrain.instance(0).stringValue(classification));
            */

        } catch (Exception ex) {
            System.out.println("ERROR: Exception caught!");
            Logger.getLogger(Id3Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
