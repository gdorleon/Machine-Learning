/* 

 * Etudiants: Gervais Sikadie et Ginel Dorleon
 * Ce programme met en oeuvre la technique du perceptron simple pour la classification de données

 */
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string.h>





using namespace std;
int nbre_classes = 0;
int nbre_cols_training = 0;
int nbre_lignes_training = 0;
int nbre_cols_test = 0;
int nbre_lignes_test = 0;

//Définition des paramètres sur les données d'entrainement et de test des bases ovarian et spam
#define NBRE_SPAM_LINE_TEST 1533
#define NBRE_SPAM_COL_TEST 59
#define NBRE_SPAM_LINE_TRN 3068
#define NBRE_SPAM_COL_TRN 59
#define NBRE_OVARIAN_LINE_TEST 84
#define NBRE_OVARIAN_COL_TEST 15156
#define NBRE_OVARIAN_LINE_TRN 169
#define NBRE_OVARIAN_COL_TRN 15156 
#define NBRE_ALLAML_LINE_TEST 34
#define NBRE_ALLAML_COL_TEST 7131
#define NBRE_ALLAML_LINE_TRN 38
#define NBRE_ALLAML_COL_TRN 7131

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
            nbre_lignes_training = NBRE_SPAM_LINE_TRN;
            nbre_cols_training = NBRE_SPAM_COL_TRN;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/spam/spam.tst") == 0) {
            nbre_lignes_test = NBRE_SPAM_LINE_TEST;
            nbre_cols_test = NBRE_SPAM_COL_TEST;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }

        if (file.compare("data/ovarian/ovarian.trn") == 0) {
            nbre_lignes_training = NBRE_OVARIAN_LINE_TRN;
            nbre_cols_training = NBRE_OVARIAN_COL_TRN;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/ovarian/ovarian.tst") == 0) {
            nbre_lignes_test = NBRE_OVARIAN_LINE_TEST;
            nbre_cols_test = NBRE_OVARIAN_COL_TEST;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }
       if (file.compare("data/leukemia/ALLAML.trn") == 0) {
            nbre_lignes_training = NBRE_ALLAML_LINE_TRN;
            nbre_cols_training = NBRE_ALLAML_COL_TRN;
            nbre_ligne = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/leukemia/ALLAML.tst") == 0) {
            nbre_lignes_test = NBRE_ALLAML_LINE_TEST;
            nbre_cols_test = NBRE_ALLAML_COL_TEST;
            nbre_ligne = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }

 // création de la matrice pour stocker nos données
         //cout << "pampam:" << nbre_ligne << endl;
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
        file_reader.close(); 
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

/*Prédiction de la classe à l'aide des inputs et des weight*/

int predire(double * weight, double *ligne) {

    int resultat_predi = 1;
    double output = 0;

    for (int i = 0; i < nbre_cols_training - 1; i++) {
        output += weight[i] * ligne[i];
    }

    if (output < 0) {
        resultat_predi = -1;
    }

    return resultat_predi;
}

//calcul des poids adaptés aux exemples
//K est le nombre maximal d'itérations
//tauxApprentissage c'est ici le taux d'apprentissage
double* perceptron(double** train_set, double tauxApprentissage, int K) {

    double *weight;
    bool bon = false;
    int count = 0;
    weight = new double[nbre_cols_training - 1];
    int classes_predites;

    //générer aléatoirement des valeurs de w 
    for (int i = 0; i < nbre_cols_training - 1; i++) {

        weight[i] = rand() / double(RAND_MAX);

    }

    //ALgorithme du perceptron simple
    for (int j = 0; j < K; j++) {
        for (int k = 0; k < nbre_lignes_training; k++) {

            classes_predites = predire(weight, train_set[k]);

            if (classes_predites != train_set[k][nbre_cols_training - 1]) {
                count++;

                for (int i = 0; i < nbre_cols_training - 1; i++) {

                    weight[i] += tauxApprentissage * (train_set[k][nbre_cols_training - 1] - classes_predites) * train_set[k][i];

                }

            }
        }
    }

    return weight;
}

//calcul de la performance de classification

double weight_taux(int **matrice_confidence) {
    double performance = 0;
    double somme_diagonale = 0;
    double somme_totale = 0;

    for (int i = 0; i < nbre_classes; i++) {
        for (int j = 0; j < nbre_classes; j++) {
            if (i == j) {
                somme_diagonale += matrice_confidence[i][j];
            }
            somme_totale += matrice_confidence[i][j];
        }
    }
    performance = somme_diagonale / somme_totale;
    return performance;

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

    //reading and storage of program's parameters
    char *trn = argv[1];
    char *tst = argv[2];
    char filename[150] = "results_perceptron_simple";
    char fichier_entrainement[50] = "";
     char txt[5] = ".txt";
    strcat(fichier_entrainement, trn);

    char fichier_test[50] = "";
    strcat(fichier_test, tst);

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
 
    test = read_donnees(fichier_test, "tst");
    training = read_donnees(fichier_entrainement, "trn");

    
    
    //Normalisation des donnees set for spam and leukemia base
    if (hold_out_file.compare("data/leukemia/ALLAML.trn") == 0 ||
            hold_out_file.compare("data/spam/spam.trn") == 0) {
 
        training = normaliser(training, nbre_lignes_training, nbre_cols_training);
        test = normaliser(test, nbre_lignes_test, nbre_cols_test);
 
    }

    //Stockage des classes
    int supposed_classes [nbre_lignes_test];
    for (int i = 0; i < nbre_lignes_test; i++) {
        supposed_classes[i] = test[i][nbre_cols_test - 1];
    }

    //Détermination du nombre de classes attendues
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

    //matrice de confusion
    int **confusion_matrix;
    confusion_matrix = new int*[nbre_classes];
    for (int j = 0; j < nbre_classes; j++) {
        confusion_matrix[j] = new int [nbre_classes];
        for (int n = 0; n < nbre_classes; n++) {
            confusion_matrix[j][n] = 0;
        }
    }

    //calcul des valeurs prédites
    double* weight;
    weight = new double[nbre_cols_training - 1];
    weight = perceptron(training, tauxApprentissage, K);

    int *prediction;
    prediction = new int[nbre_lignes_test];
    for (int i = 0; i < nbre_lignes_test; i++) {
        prediction[i] = predire(weight, test[i]);
    }

    //Matrice de confusion
    for (int m = 0; m < nbre_lignes_test; m++) {

        int pos1 = std::find(classes.begin(), classes.end(), (int) test[m][nbre_cols_test - 1]) - classes.begin();
        int pos2 = std::find(classes.begin(), classes.end(), (int) prediction[m]) - classes.begin();
        confusion_matrix[pos1][pos2]++;
    }

    //Nombre de classes matrice de confusion et examples
    cout << "nombre de classes :" << nbre_classes << endl;

    for (int a = 0; a < nbre_classes; a++) {
        for (int b = 0; b < nbre_classes; b++) {
            cout << confusion_matrix[a][b] << " ";
        }
        cout << endl;
    }

    cout << "le taux de précision est :" << weight_taux(confusion_matrix)*100 << "%" << endl;

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
       

        fichier_perceptron_simple_Resultat << "Résultats de l'apprentissage avec le Perceptron simple \n" << endl;

        fichier_perceptron_simple_Resultat << " Méthode utilisée: Perceptron simple. \n "  << endl;
        fichier_perceptron_simple_Resultat << " Taux d'apprentissage : " << tauxApprentissage  << endl;
        fichier_perceptron_simple_Resultat << " Nombre maximal d'itérations : " << K << endl;
        fichier_perceptron_simple_Resultat << " Taux de weight placement global: " << weight_taux(confusion_matrix)*100 << "% \n";

        fichier_perceptron_simple_Resultat << "Matrice de confusion \n";



        for (int i = 0; i < nbre_classes; i++) {
        
            for (int j = 0; j < nbre_classes; j++) {
                fichier_perceptron_simple_Resultat  << confusion_matrix[i][j] << "  " ; 
            }
          
            fichier_perceptron_simple_Resultat << "\n"<< endl;
        }
 
        fichier_perceptron_simple_Resultat.close();
    } else
        cerr << "Impossible d'ouvrir le fichier .txt des résultats!" << endl;


    return 0;

}
