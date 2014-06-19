/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package c4.pkg5crossv;

import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * klasa odczytujaca format ARFF
 * @author si
 */
public class DataLoad {
    public static Instances loadData(String fileName)
    throws IOException
    {
        ArffLoader loader = new ArffLoader(); //Utworzenie obiektu czytajacego dane z formatu ARFF
        loader.setFile(new File(fileName)); //Ustawienie pliku do odczytania
        return loader.getDataSet(); //Odczytanie danych z pliku
    }
}
