/* 
 * Etudiants: Gervais Sikadie et Ginel Dorleon
 */

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string.h>

//Définition des paramètres sur les données d'entrainement et de test des bases ovarian et spam
#define NBRE_SPAM_ligne_TRN 3068
#define NBRE_SPAM_COL_TRN 59
#define NBRE_SPAM_ligne_TEST 1533
#define NBRE_SPAM_COL_TEST 59
#define NBRE_OVARIAN_ligne_TRN 169
#define NBRE_NBRE_OVARIAN_COL_TEST 15156
#define NBRE_OVARIAN_ligne_TEST 84
#define NBRE_OVARIAN_COL_TEST 15156
#define NBRE_ALLAML_ligne_TEST 34
#define NBRE_ALLAML_COL_TEST 7131
#define NBRE_ALLAML_ligne_TRN 38
#define NBRE_ALLAML_COL_TRN 7131
using namespace std;

int nbre_classes = 0;
int nbre_cols_training = 0;
int nbre_lignes_training = 0;
int nbre_cols_test = 0;
int nbre_lignes_test = 0;

//fonction de lecture de données dans un fichier et retour du tableaux à deux dimension de données lues
double **read_donnees(string file, string type) {
    int nbre_ligne = 0, nb_col = 0;
    double **donnees;
    ifstream file_reader(file.c_str(), ios::in);
    if (file_reader)
    {
        
         //initialiser le nombre de ligne et de colonnes des données training et test selon qu'on utilise
        if (file.compare("data/spam/spam.trn") != 0 && file.compare("data/ovarian/ovarian.trn") != 0 && file.compare("data/leukemia/ALLAML.trn") != 0
                && type.compare("trn") == 0) {
            file_reader >> nbre_lignes_training >> nbre_cols_training;
            nbre_cols_training++;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/spam/spam.tst") != 0 && file.compare("data/ovarian/ovarian.tst") != 0 && file.compare("data/leukemia/ALLAML.tst") != 0
                && type.compare("tst") == 0) {
            file_reader >> nbre_lignes_test >> nbre_cols_test;
            nbre_cols_test++;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;
        }

        if (file.compare("data/spam/spam.trn") == 0) {
            nbre_lignes_training = NBRE_SPAM_ligne_TRN;
            nbre_cols_training = NBRE_SPAM_COL_TRN;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/spam/spam.tst") == 0) {
            nbre_lignes_test = NBRE_SPAM_ligne_TEST;
            nbre_cols_test = NBRE_SPAM_COL_TEST;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }

        if (file.compare("data/ovarian/ovarian.trn") == 0) {
            nbre_lignes_training = NBRE_OVARIAN_ligne_TRN;
            nbre_cols_training = NBRE_NBRE_OVARIAN_COL_TEST;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/ovarian/ovarian.tst") == 0) {
            nbre_lignes_test = NBRE_OVARIAN_ligne_TEST;
            nbre_cols_test = NBRE_OVARIAN_COL_TEST;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }
	if (file.compare("data/leukemia/ALLAML.trn") == 0) {
            nbre_lignes_training = NBRE_ALLAML_ligne_TRN;
            nbre_cols_training = NBRE_ALLAML_COL_TRN;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/leukemia/ALLAML.tst") == 0) {
            nbre_lignes_test = NBRE_ALLAML_ligne_TEST;
            nbre_cols_test = NBRE_ALLAML_COL_TEST;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }
 // création de la matrice pour stocker nos données
   
        donnees = new double* [nbre_ligne];
        for (int i = 0; i < nbre_ligne; i++) {
            donnees[i] = new double[nb_col];
        }

       // on met l'input par défaut à 1 dans toutes les premières colonnes du tableau en parcourant chaque ligne
        
        for (int i = 0; i < nbre_ligne; i++) {
            donnees[i][0] = 1;
        }
        
          //stockage des donnees dans la matrice
        for (int k = 0; k < nbre_ligne; k++) {
            for (int l = 1; l < nb_col; l++) {
                file_reader >> donnees[k][l];              
            }
        }
        
        file_reader.close(); // ferméture du fichier
    } else {
        cerr << "Open file error" << endl;
    }

    return donnees;
}

//comparer deux lignes
bool comparedeuxLignes(double* firstRow, double* secondRow) {
    return (firstRow[1] < secondRow[1]);
}

//normalisation des données
double ** normalize_donnees(double** donnees, int nbre_ligne, int nb_col){
    
   double* column;
   column = new double[nbre_ligne];
   
   for(int i=0; i< nb_col-1; i++){
        for(int j=0; j< nbre_ligne; j++){           
            column[j] = donnees[j][i];            
        }
     std::sort(column, column + nbre_ligne); 
     for(int j=0; j< nbre_ligne; j++){
         
         donnees[j][i] = (donnees[j][i]-column[0])/(column[nbre_ligne-1]-column[0]);
                 
     }     
   }
   
   return donnees;
}

/*Prédiction de la classe à l'aide des inputs et des weight*/
double predictor(double * weight, double *ligne){
    
   
    double sortie =0;
    
    for(int i =0; i < 3; i++){
        
        sortie += weight[i]*ligne[i];
        
    }
    sortie = 1/(1+exp(-1*sortie));
    
  
    return sortie;
}

//Propager la valeur de l'input dans le réseaux de neuronens
double *propager(double **matrix_weight, double *ligne){
    
    double *out;
    out = new double[4];
    out[0] = 1;
    
    for(int i = 1; i< 3; i++){
        out[i] = predictor(matrix_weight[i-1], ligne);
    }
    
        out[3] = predictor(matrix_weight[2],out);
    
        return out;
}

//retro propager
double **retro_propager(double *out, double **matrix_weight, double *ligne, double tauxApprentissage){
    
    double *delta;
    double so1;
    delta = new double[9];
    double ** matrice_ajour;
    matrice_ajour = new double*[3];
    for(int i =0; i < 3; i++){
        matrice_ajour[i] = new double[3];
    }
    
    //computing delta values for all weigth and biais
    so1 = -1*(ligne[nbre_cols_training-1]-out[3])* out[3]*(1-out[3]);
    delta[5] = so1*out[2];
    delta[4] = so1*out[1];
    delta[3] = so1*matrix_weight[2][2]*out[2]*(1-out[2])*ligne[2];
     delta[2] = so1*matrix_weight[2][2]*out[2]*(1-out[2])*ligne[1];
      delta[1] = so1*matrix_weight[2][1]*out[1]*(1-out[1])*ligne[2];
       delta[0] = so1*matrix_weight[2][1]*out[1]*(1-out[1])*ligne[1];
       
       delta[6] = so1;//biais 3
       delta[7] = so1*matrix_weight[2][2]*out[2]*(1-out[2]); // biais 2
       delta[8] =  so1*matrix_weight[2][1]*out[1]*(1-out[1]); // biais 1
    
       //update weight and biais
    matrice_ajour[2][2] = matrix_weight[2][2]-tauxApprentissage*delta[5];
    matrice_ajour[2][1] = matrix_weight[2][1] - tauxApprentissage*delta[4];
    matrice_ajour[1][2] = matrix_weight[1][2] - tauxApprentissage*delta[3];
     matrice_ajour[1][1] = matrix_weight[1][1] - tauxApprentissage*delta[2];
    matrice_ajour[0][2] = matrix_weight[0][2] - tauxApprentissage*delta[1];
    matrice_ajour[0][1] = matrix_weight[0][1] - tauxApprentissage*delta[0];
    matrice_ajour[2][0] = matrix_weight[2][0] - tauxApprentissage*delta[6];//biais 3
    matrice_ajour[1][0] = matrix_weight[1][0] - tauxApprentissage*delta[7];// biais 2
    matrice_ajour[0][0] = matrix_weight[0][0] - tauxApprentissage*delta[8];//biais 1
    
    
    return matrice_ajour;
}

//calcul du poid correspondant a l'ensemble d'entrainement
double** perceptron_multi(double** train_set, double tauxApprentissage, int K) {
    
    double **weight;
    weight = new double*[3];
    for(int i =0; i < 3; i++){
        weight[i] = new double[3];
    }
    
     //donner le valeus aléatoires aux poids
    for(int i =0; i < 3; i++ ){
        for(int j =0; j < 3; j++ ){        
       weight[i][j] = rand() / double(RAND_MAX); 
        }
    }
    
    //retropropager
    for(int j= 0; j < K ; j++){
        for(int k =0; k< nbre_lignes_training; k++){                    
          weight = retro_propager(propager(weight,train_set[k]),weight,train_set[k],tauxApprentissage) ;                          
        }        
    }
    
    return weight;
}

//calucul de la performance de classification
double good_rate(int **conf_mat) {
    double taux = 0;
    double somme_diag = 0;
    double somme_tot = 0;

    for (int i = 0; i < nbre_classes; i++) {
        for (int j = 0; j < nbre_classes; j++) {
            if (i == j) {
                somme_diag += conf_mat[i][j];
            }
            somme_tot += conf_mat[i][j];
        }
    }
    taux = somme_diag / somme_tot;
    return taux;

}
//normalisation des données

double ** normaliser(double** donnees, int nbre_ligne, int nb_col) {

    double* column;
    column = new double[nbre_ligne];

    for (int i = 1; i < nb_col - 1; i++) {
        for (int j = 0; j < nbre_ligne; j++) {

            column[j] = donnees[j][i];

        }
        std::sort(column, column + nbre_ligne);
        for (int j = 0; j < nbre_ligne; j++) {

            donnees[j][i] = (donnees[j][i] - column[0]) / (column[nbre_ligne - 1] - column[0]);

        }
    }

    return donnees;
}
//Hold-out operation

void splitfile(string file) {

    //hold-out of spam base
    if (file.compare("data/spam/spam.trn") == 0) {
        system("sort -R data/spam/spam.data > data/spam/spam.txt");
        system("head -n3068 data/spam/spam.txt > data/spam/spam.trn");
        system("tail -n1533 data/spam/spam.txt > data/spam/spam.tst");

    } else { //hold-out of ovarian base

        system("sort -R data/ovarian/ovarian.data > data/ovarian/ovarian.txt");
        system("head -n169 data/ovarian/ovarian.txt > data/ovarian/ovarian.trn");
        system("tail -n84 data/ovarian/ovarian.txt > data/ovarian/ovarian.tst");
    }

}
int main(int argc, char** argv) {
    
    //lecture et stockage des paramtères du programme
    char *trn = argv[1];
    char *tst = argv[2];

    char filename[150] = "results_perceptron_multi";

   char fichier_entrainement[50] = "";
     char txt[5] = ".txt";
    strcat(fichier_entrainement, trn);

    char test_file[50] = "";
    strcat(test_file, tst);
    
    double tauxApprentissage;
    tauxApprentissage = atof(argv[3]);
    int K;
    K = atoi(argv[4]);


    double** training;
    double** test;

   //Lire les donnees de la base
    string hold_out_file(fichier_entrainement);
    if (hold_out_file.compare("data/ovarian/ovarian.trn") == 0 ||
            hold_out_file.compare("data/spam/spam.trn") == 0) {
        splitfile(fichier_entrainement);
    }
    //Reading of donnees from specific base
    test = read_donnees(test_file, "tst");
    training = read_donnees(fichier_entrainement, "trn");
   

 
    //Normalisation des donnees set for spam and leukemia base
    if (hold_out_file.compare("data/leukemia/ALLAML.trn") == 0 ||
            hold_out_file.compare("data/spam/spam.trn") == 0) {
 
        training = normaliser(training, nbre_lignes_training, nbre_cols_training);
        test = normaliser(test, nbre_lignes_test, nbre_cols_test);
 
    }

     //Storage of expected classes
    int supposed_classes [nbre_lignes_test];
    for (int i = 0; i < nbre_lignes_test; i++) {
        supposed_classes[i] = test[i][nbre_cols_test - 1];
    }

     //Determination of number of expected classes
    std::sort(supposed_classes, supposed_classes + nbre_lignes_test);
    int compt = 0;
    std::vector<int> classes;
    for (int i = 0; i < nbre_lignes_test; i++) {

        compt = std::count(supposed_classes, supposed_classes +
                nbre_lignes_test, supposed_classes[i]);
        classes.push_back(supposed_classes[i]);
        i = i + compt - 1;
        nbre_classes++;

    }

   //creation of the confusion matrix
    int **matrice_confusion;
    matrice_confusion = new int*[nbre_classes];
    for (int j = 0; j < nbre_classes; j++) {
        matrice_confusion[j] = new int [nbre_classes];
        for (int n = 0; n < nbre_classes; n++) {
            matrice_confusion[j][n] = 0;
        }
    }
    
    //computing of predited classes
    double** weight;
    weight = new double*[3];
    for(int i =0; i < 3; i++){
        weight[i] = new double[3];
    }
    weight = perceptron_multi(training, tauxApprentissage, K);
    
    int *prediction;
    prediction = new int[nbre_lignes_test];


    for (int i = 0; i < nbre_lignes_test; i++) {
        double* a = propager(weight,test[i]);
        if(a[3]>0.5){
        prediction[i] = 1;}
        else{ prediction[i] =0;
        }        
    }
    
    //Matrice de confusiion
    for (int m = 0; m < nbre_lignes_test; m++) {
        int pos1 = std::find(classes.begin(), classes.end(), (int) test[m][nbre_cols_test - 1]) - classes.begin();
        int pos2 = std::find(classes.begin(), classes.end(), (int) prediction[m]) - classes.begin();
        matrice_confusion[pos1][pos2]++;
    }

     //Nompre de classes et matrice de confusion affichage
    cout << "nombre de classes :" << nbre_classes << endl;
    
    for (int a = 0; a < nbre_classes; a++) {
        for (int b = 0; b < nbre_classes; b++) {
            cout << matrice_confusion[a][b] << " ";
        }
        cout << endl;
    }

    cout << "le taux de précision est :" << good_rate(matrice_confusion)*100 << "%" << endl;

   char* s;
   s = strtok (argv[1] ,"/");

	
	s = strtok (NULL, "/");
       // s = strtok (NULL, "/");
   strcat(filename,s);  
   strcat(filename,argv[3]);
   strcat(filename,argv[4]);
   strcat(filename,txt);
    //Writting of result in an txt file
    ofstream fichier_perceptron_simple_Resultat(filename, ios::out | ios::trunc);

    if (fichier_perceptron_simple_Resultat) {
  
        fichier_perceptron_simple_Resultat << "Résultats de l'apprentissage avec le Perceptron multi-couches \n" << endl;

        fichier_perceptron_simple_Resultat << "Méthode utilisée: Perceptron multi-couches. " << "" << endl;
        fichier_perceptron_simple_Resultat << "Pas d'apprentissage : " << tauxApprentissage << "" << endl;
        fichier_perceptron_simple_Resultat << "Nombre maximal d'itérations : " << K << "" << endl;
        fichier_perceptron_simple_Resultat << "Taux de bon placement global: " << good_rate(matrice_confusion)*100 << "%\n";

        fichier_perceptron_simple_Resultat << "Matrice de confusion ";
      
        for (int i = 0; i < nbre_classes; i++) {
         
            for (int j = 0; j < nbre_classes; j++) {
                fichier_perceptron_simple_Resultat << matrice_confusion[i][j] << "  " << "\n"; 
            }
            fichier_perceptron_simple_Resultat << "\n ";
            fichier_perceptron_simple_Resultat << endl;
        }//end for
   
        fichier_perceptron_simple_Resultat.close();
    } else
        cerr << "Impossible d'ouvrir le fichier .txt" << endl;

    return 0;

}
