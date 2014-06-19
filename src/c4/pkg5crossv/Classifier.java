/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package c4.pkg5crossv;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author si
 */
public class Classifier {
    public static void C45() throws FileNotFoundException, IOException, Exception{
        Instances data = DataLoad.loadData("./src/data/irysy.arff");

        //Ustawienie atrybutu decyzyjnego (ostatni atrybut)
        data.setClassIndex(data.numAttributes() - 1);

        //OPCJE:
        //-U -> budowa drzewa bez przycinania (ostre liscie)
        //-C -> <wspolczynnik dokladnosci> - ustawienie wspolczynnika dokladnosci dla lisci (default 0.25)
        //-M -> ustawienie minimalnej liczby obiektow w lisciu dla ktorej lisc nie jest dzielony (default 2)

        //Ustalenie opcji
        String[] options = Utils.splitOptions("-U -M 10");
        
        J48 tree = new J48();    
        tree.setOptions(options); //Ustawienie opcji
        tree.buildClassifier(data);  // Tworzenie klasyfikatora (drzewa)

        System.out.println(tree.toString()); //Wypisanie drzewa w formie tekstowej
        
        System.out.println("TRAIN&TEST");
        trainAndTest();
    }
    
    public static void trainAndTest()throws FileNotFoundException, IOException, Exception{
        
        Instances data = DataLoad.loadData("./src/data/irysy.arff");
        data.setClassIndex(data.numAttributes() - 1);

        //Losowy podzial tablicy
        data.randomize(new Random());
        double percent = 60.0;
        int trainSize = (int) Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        String[] options = Utils.splitOptions("-U -M 10");
        J48 tree = new J48();
        tree.setOptions(options);
        tree.buildClassifier(trainData);
        
        Evaluation eval2 = new Evaluation(trainData);
        eval2.crossValidateModel(tree, testData, 10, new Random(1)); // 5 - fold
        System.out.println(eval2.toSummaryString("Wyniki:", false)); //Wypisanie testovania cross validation
    }
    
}
