
/* 
 * 
 * Etudiants: Fotsing Sikadie Gervais et Ginel Dorleon Promotion 20
 *
 * Ceci est une implémentation de l'algorithme k-nn qui prend en entrée un ensemble de données d'apprentissage et un ensemble de données tests pour calculer la matrice de confusion
 */
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string.h>





using namespace std;

int nbre_classes = 0;
int nbre_cols_training = 0; 
int nbre_lignes_training = 0;
int nbre_cols_test = 0;
int nbre_lignes_test = 0;
//Paramètres par défaut pour le protocole hold-out
#define NBRE_FP_LIGNE_TRN 320 // représente 2/3 des données totales(2*490/3 dans le cas présent)
#define NBRE_FP_LIGNE_TEST 160 // représente 1/3 des données totales(490 individus au total dans l'exemple de l'exercice)
#define NBRE_FP_ATTR 2001 //nombre d'attributs + 1 du jeux de données


//fonction de lecture de données dans un fichier et retour du tableaux à deux dimension de données lues

float **read_data(string file, string type) {
    int nb_line = 0, nb_col = 0;
    float **donnees;
    ifstream lecteur_fichier(file.c_str(), ios::in);
    if (lecteur_fichier) 
    {

        //initialiser le nombre de ligne et de colonnes des données training et test selon qu'on utilise le protocole hold-out ou pas
        if (file.compare("data/fp/fp.trn") == 0) {
            nbre_lignes_training = NBRE_FP_LIGNE_TRN;
            nbre_cols_training = NBRE_FP_ATTR;
            nb_line = nbre_lignes_training;
            nb_col = nbre_cols_training;
        }

        if (file.compare("data/fp/fp.tst") == 0) {
            nbre_lignes_test = NBRE_FP_LIGNE_TEST;
            nbre_cols_test = NBRE_FP_ATTR;
            nb_line = nbre_lignes_test;
            nb_col = nbre_cols_test;

        }       

        // création de la matrice pour stocker nos données
        donnees = new float* [nb_line];
        for (int i = 0; i < nb_line; i++) {
            donnees[i] = new float[nb_col];
        }

        //stockage des données du fichier dans le tableau
        for (int k = 0; k < nb_line; k++) {
            for (int l = 0; l < nb_col; l++) {
                lecteur_fichier >> donnees[k][l];
            }

        }


        lecteur_fichier.close(); 
    } else {
        cerr << "fichier on existant erreur!" << endl;
    }

    return donnees;
}


//fonction de lecture de données dans un fichier et retour du tableaux à deux dimension de données lues

float **read_data_classic(string file, string type,int a,int b,int c) {
    int nb_line = 0, nb_col = 0;
    float **donnees;
    ifstream lecteur_fichier(file.c_str(), ios::in);
    if (lecteur_fichier) 
    {
     
        if (file.compare("data/fp/fp.trn") != 0
                && type.compare("trn") == 0) {
            //lecteur_fichier >> nbre_lignes_training >> nbre_cols_training;
            nbre_lignes_training=a;
            nbre_cols_training=1+b;
            //nb_line = nbre_lignes_training;
            //nb_col = nbre_cols_training;
              nb_line = nbre_lignes_training;
              nb_col = nbre_cols_training;
        }

        if (file.compare("data/fp/fp.tst") != 0
                && type.compare("tst") == 0) {
            //lecteur_fichier >> nbre_lignes_test >> nbre_cols_test;
            //nb_line = nbre_lignes_test;
            //nb_col = nbre_cols_test;
              nbre_lignes_test=c;
              nbre_cols_test=1+b;
              nb_line = nbre_lignes_test;
              nb_col = nbre_cols_test; 
        }
        // création de la matrice pour stocker nos données
        donnees = new float* [nb_line];
        for (int i = 0; i < nb_line; i++) {
            donnees[i] = new float[nb_col];
        }

        //stockage des données du fichier dans le tableau
        for (int k = 0; k < nb_line; k++) {
            for (int l = 0; l < nb_col; l++) {
                lecteur_fichier >> donnees[k][l];
            }

        }


        lecteur_fichier.close(); 
    } else {
        cerr << "fichier on existant erreur!" << endl;
    }

    return donnees;
}


//distance euclidienne d'une individu à un autre

float dist_euclidienne(float individu1[], float individu2[]) {

    float distance = 0;
    for (int i = 0; i < nbre_cols_training - 1; i++) {
        distance += (individu1[i] - individu2[i])*(individu1[i] - individu2[i]);
    }
    distance = sqrt(distance);
    return distance;
}

//Distance de manathan entre deux vecteurs individu1 et individu2

float dist_manathan(float individu1[], float individu2[]) {

    float distance = 0;
    for (int i = 0; i < nbre_cols_training - 1; i++) {
        distance += abs((individu1[i] - individu2[i]));
    }
    return distance;
}

//Distance cosinus entre deux vecteurs individu1 et individu2

float dist_cosinus(float individu1[], float individu2[]) {

    float distance,norme1,norme2,scalaire;
    distance=0;
    norme1=0;
    norme2=0;
    scalaire=0;
    for (int i = 0; i < nbre_cols_training - 1; i++) {
        norme1 += pow(individu1[i],2);
    }
    norme1=sqrt(norme1);
    for (int i = 0; i < nbre_cols_training - 1; i++) {
        norme2 += pow(individu2[i],2);
    }
    norme2=sqrt(norme2);

    for (int i = 0; i < nbre_cols_training - 1; i++) {
        scalaire += individu1[i]*individu2[i];
    }
    norme2=sqrt(norme2);
    distance=norme1*norme2/scalaire;
    return distance;
}

//Calcul de la distance d'un vecteur à tous les autres individus

float** dist_vec_to_ensemble(float** ensemble, float vector[],int d) {

    float** dist;
    dist = new float*[nbre_lignes_training];
    for (int i = 0; i < nbre_lignes_training; i++) {
        dist[i] = new float[2];
    }

    for (int i = 0; i < nbre_lignes_training; i++) {
        dist[i][0] = i;
        if(d==1)
        dist[i][1] = dist_euclidienne(ensemble[i], vector);
        else if(d==2)
        dist[i][1] = dist_manathan(ensemble[i], vector);
        else if(d==3)
        dist[i][1] = dist_cosinus(ensemble[i], vector);
        else  dist[i][1] = dist_euclidienne(ensemble[i], vector);

    }
    return dist;
}


//comparer deux lignes

bool compareTwoRows(float* rowA, float* rowB) {
    return (rowA[1] < rowB[1]);
}

//Algorithme des k-mean pour la prédiction de classe

float k_means(float** ensemble, float vector[], int k, int d) {

    //calcul de la distance entre un vecteur et tous les autres vecteurs 
    float ** tabdist = dist_vec_to_ensemble(ensemble, vector, d);

    //tri du tableau obtenu par ordre croissant de distance
    std::sort(tabdist, tabdist + nbre_lignes_training, &compareTwoRows);

    //stockage des k premières classes obtenues
    float* knn;
    knn = new float[k];

    for (int i = 0; i < k; i++) {
        knn[i] = ensemble[(int) tabdist[i][0]][nbre_cols_training - 1];
    }
    std::sort(knn, knn + k);

    //Calcul du nombre de classes
    int * compteur;
    compteur = new int[k];
    for (int i = 0; i < k; i++) {
        compteur[i] = 0;
    }

    for (int i = 0; i < k; i++) {

        compteur[i] = std::count(knn, knn + k, knn[i]);
        i = i + compteur[i] - 1;
    }

    //Détermination de la classe dominante
    int ind_maximal = 0;
    for (int m = 1; m < k; m++) {
        if (compteur[ind_maximal] < compteur[m]) {
            ind_maximal = m;
        }
    }
    float classe = knn[ind_maximal];

    return classe;
}

//calculer le taux de bon classement matrice de confusion

float good_ranking_rate(int **matrice_confusion) {
    float taux = 0;
    float diag_sum = 0;
    float total_sum = 0;

    //division de tous les éléments situés à la diagonale de la matrice par la somme de tous les éléments de la matrice
    for (int i = 0; i < nbre_classes; i++) {
        for (int j = 0; j < nbre_classes; j++) {
            if (i == j) {
                diag_sum += matrice_confusion[i][j];
            }
            total_sum += matrice_confusion[i][j];
        }
    }
    taux = diag_sum / total_sum;
    return taux;

}

//Lorsque le protocole Hold-out est choisit, cette fonction  permet de segmenter les données en proportion 1/3 test(fp.trn) et 2/3 training(fp.tst)

void splitfile() {
//execution des commandes shell pour spliter le fichier en deux parties
    system("sort -R data/fp/fp.data > data/fp/fp.txt");// le random sort nous permet de mélanger un fichier de manière aléatoire ceci à pour but de consituer des ensembles d'entraitement et des ensembles tests tout à fait différents afin de choisir le meuilleur ensemble d'apprentissage
    system("head -n320 data/fp/fp.txt > data/fp/fp.trn");
    system("tail -n160 data/fp/fp.txt > data/fp/fp.tst");
}

int main(int argc, char** argv) {

    //lecture et stockage des paramètres du programme

    char *tst = argv[2];
    char *trn = argv[1];
    char trn_string[50] = "";
     char txt[5] = ".txt";
    strcat(trn_string, trn);
    char filename[150] = "results";
    char test_file[50] = "";
    strcat(test_file, tst);
    int k;
    k = atoi(argv[3]);
    int d= atoi(argv[4]);

    float** trainingdata;
    float** testdata;


    //Lecture des données du jeux de données destiné à appliquer le protocole hold-out au cas où on veut appliquer le protocole hold-out
    string fichier_hold(trn_string); // j'affecte le trn_string nom du fichier à fichier_hold
    if (fichier_hold.compare("data/fp/fp.trn") == 0) {
        splitfile();
        //chargement des données d'entrainement et de test
        trainingdata = read_data(trn_string, "trn");
        testdata = read_data(test_file, "tst");

    }else{
   //chargement des données d'entrainement et de test
    trainingdata = read_data_classic(trn_string, "trn",atoi(argv[5]),atoi(argv[6]),atoi(argv[7]));
    testdata = read_data_classic(test_file, "tst",atoi(argv[5]),atoi(argv[6]),atoi(argv[7]));

    }

    //Stockage des résultats de classes attendues pour le test
    int classes_attendues [nbre_lignes_test];
    for (int i = 0; i < nbre_lignes_test; i++) {
        classes_attendues[i] = testdata[i][nbre_cols_test - 1];
    }

    //Calcul du nombre total de classes
    std::sort(classes_attendues, classes_attendues + nbre_lignes_test);
    int compt = 0;
    std::vector<int> classes;
    for (int i = 0; i < nbre_lignes_test; i++) {

        compt = std::count(classes_attendues, classes_attendues +
                nbre_lignes_test, classes_attendues[i]);
        classes.push_back(classes_attendues[i]);
        i = i + compt - 1;
        nbre_classes++;

    }

    //création de la matrice de confusion
    int **confusion_matrix;
    confusion_matrix = new int*[nbre_classes];
    for (int j = 0; j < nbre_classes; j++) {
        confusion_matrix[j] = new int [nbre_classes];
        for (int n = 0; n < nbre_classes; n++) {
            confusion_matrix[j][n] = 0;
        }
    }

    //passage du vecteur au modèle
    int *prediction;
    prediction = new int[nbre_lignes_test];
    for (int i = 0; i < nbre_lignes_test; i++) {
        prediction[i] = (int) k_means(trainingdata, testdata[i], k,d);
    }

    //calcul de la matrice de confusion
    for (int m = 0; m < nbre_lignes_test; m++) {
        int pos1 = std::find(classes.begin(), classes.end(), (int) testdata[m][nbre_cols_test - 1]) - classes.begin();
        int pos2 = std::find(classes.begin(), classes.end(), (int) prediction[m]) - classes.begin();
        confusion_matrix[pos1][pos2]++;
    }

    //affichage du nombre de classes, de la matrice de confusion et de la performance
    cout << "class number : " << nbre_classes << endl;

    for (int a = 0; a < nbre_classes; a++) {
        for (int b = 0; b < nbre_classes; b++) {

            cout << confusion_matrix[a][b] << " ";
        }
        cout << endl;

    }

    cout << "the rate of accuracy is :" << good_ranking_rate(confusion_matrix)*100 << " %" << endl;
char* s;
s = strtok (argv[1] ,"/");

	
	s = strtok (NULL, "/");
       // s = strtok (NULL, "/");
strcat(filename,s);  
strcat(filename,argv[3]);
strcat(filename,argv[4]);
strcat(filename,txt);
    //Writting of result in an txt file
    ofstream fichierKNNResults(filename, ios::out | ios::trunc); 

    if (fichierKNNResults) {
    
        fichierKNNResults << "Classification results with KNN \n";
     


        fichierKNNResults << " Method used: KNN.  \n k value: " << k << "\n " << endl;

        fichierKNNResults << "global rate of good classification: " << good_ranking_rate(confusion_matrix)*100 << "% \n";

        fichierKNNResults << " confusion Matrix \n";
 


        for (int i = 0; i < nbre_classes; i++) {
           
            for (int j = 0; j < nbre_classes; j++) {
                fichierKNNResults << " " << confusion_matrix[i][j] ;
            }
            fichierKNNResults << "\n";
            fichierKNNResults << endl;
        }

        fichierKNNResults << "";
        fichierKNNResults << "";
        fichierKNNResults.close();
         //cout << "coordonnées!!!" << nbre_lignes_training<<"   " << nbre_cols_training << endl;
    }
    else
        cerr << "Impossible d'ouvrir le fichier txt des résultats!!!" << nbre_lignes_training << nbre_cols_training << endl;

    return 0;

}
