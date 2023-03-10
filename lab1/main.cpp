#include <iostream>
# include <cmath>
using namespace std;

float f_float(float x){
    return sqrt(pow(x, 2.F) + 1.F) - 1.F;
}

float g_float(float x){
    return pow(x, 2.F) / (sqrt(pow(x, 2.F) + 1.F) + 1.F);
}

double f_double(double x){
    return sqrt(pow(x, 2) + 1) - 1;
}

double g_double(double x){
    return pow(x, 2) / (sqrt(pow(x, 2) + 1) + 1);
}

long double f_long_double(long double x){
    return sqrt(pow(x, 2) + 1) - 1;
}

long double g_long_double(long double x){
    return pow(x, 2) / (sqrt(pow(x, 2) + 1) + 1);
}

void get_result_float_neg(){
    for(int i = 1; i <= 50; i++){
        float x = f_float(pow(8.F, -float(i)));
        float y = g_float(pow(8.F, -float(i)));
        cout << "Krok " << i << ": "<<  "Wynik dla f(x): " << x << ", Wynik dla g(x): " << y << endl
        <<"Roznica wynikow f(x) - g(x): "<< x - y << ", Roznica wynikow g(x) - f(x): " << y - x << endl;
    }
}

void get_result_double_neg(){
    for(int i = 1; i <= 300; i++){
        double x = f_double(pow(8, -i));
        double y = g_double(pow(8, -i));
        cout << "Krok " << i << ": "<<  "Wynik dla f(x): " << x << ", Wynik dla g(x): " << y << endl
             <<"Roznica wynikow f(x) - g(x): "<< x - y << ", Roznica wynikow g(x) - f(x): " << y - x << endl;
    }
}

void get_result_long_double_neg(){
    for(int i = 0; i <= 360; i++){
        long double x = f_long_double(pow(8, -i));
        long double y = g_long_double(pow(8, -i));
        cout << "Krok " << i << ": "<< "Wynik dla f(x): " << x << ", Wynik dla g(x): " << y << endl
             <<"Roznica wynikow f(x) - g(x): "<< x - y << ", Roznica wynikow g(x) - f(x): " << y - x << endl;
    }
}

void get_result_float_pos(){
    for(int i = 1; i <= 50; i++){
        float x = f_float(pow(8.F, float(i)));
        float y = g_float(pow(8.F, float(i)));
        cout << "Krok " << i << ": "<<  "Wynik dla f(x): " << x << ", Wynik dla g(x): " << y << endl
             <<"Roznica wynikow f(x) - g(x): "<< x - y << ", Roznica wynikow g(x) - f(x): " << y - x << endl;
    }
}

void get_result_double_pos(){
    for(int i = 1; i <= 180; i++){
        double x = f_double(pow(8, i));
        double y = g_double(pow(8, i));
        cout << "Krok " << i << ": "<<  "Wynik dla f(x): " << x << ", Wynik dla g(x): " << y << endl
             <<"Roznica wynikow f(x) - g(x): "<< x - y << ", Roznica wynikow g(x) - f(x): " << y - x << endl;
    }
}

void get_result_long_double_pos(){
    for(int i = 1; i <= 345; i++){
        long double x = f_long_double(pow(8, i));
        long double y = g_long_double(pow(8, i));
        cout << "Krok " << i << ": "<< "Wynik dla f(x): " << x << ", Wynik dla g(x): " << y << endl
             <<"Roznica wynikow f(x) - g(x): "<< x - y << ", Roznica wynikow g(x) - f(x): " << y - x << endl;
    }
}

int main() {
    //get_result_float_neg();
    //get_result_double_neg();
    //get_result_long_double_neg();
    get_result_float_pos();
    //get_result_double_pos();
    //get_result_long_double_pos();
    return 0;
}
