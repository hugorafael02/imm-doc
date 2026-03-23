/**
 * @brief Algoritmo principal IMM (Influence Maximization via Martingales).
 * * Implementa o framework de amostragem teórica em duas fases para o problema de Maximização de Influência.
 * A função determina o número exato de Reverse Reachable Sets (RR-Sets) necessários para garantir um
 * limite de aproximação de (1 - 1/e - epsilon) com alta probabilidade matemática.
 * * @param V Número total de vértices no grafo.
 * @param rs Grafo transposto (Reverse Structure), onde rs[v] contém as arestas que apontam para v.
 * @param model Modelo de propagação ("ic" ou "lt").
 * @param k Tamanho do orçamento (número de sementes a serem selecionadas).
 * @param eps Parâmetro de erro epsilon ($ \epsilon $). Define a tolerância de erro da aproximação.
 * @param ell Parâmetro de confiança probabilística ($ \ell $). A garantia de sucesso é $ 1 - 1/V^\ell $.
 * @param candidates Máscara booleana de tamanho V. Se candidates[v] == false, o nó não pode ser escolhido como semente.
 * * @return std::vector<int> Conjunto final contendo os IDs das k sementes ótimas.
 */
vector<int> imm(int V, vector<vector<edge> >& rs, string model, int k, double eps, double ell,
	const vector<bool>& candidates) {
	
	// ========================================================================
	// PREPARAÇÃO DE CONSTANTES TEÓRICAS
	// ========================================================================
	const double e = exp(1);
	
	/// log_VCk calcula o logaritmo natural da combinação $ \log \binom{V}{k} $.
	/// Usado para definir o espaço de busca combinatório das sementes.
	double log_VCk = log_nCk(V, k);

	/// Ajuste empírico/teórico do parâmetro $ \ell $ para estabilidade assintótica.
	ell = ell * (1 + log(2) / log(V));
	
	/// Cálculo de epsilon linha ($ \epsilon' $), usado especificamente na Fase 1 (Estimação).
	double eps_p = sqrt(2) * eps;

	cout << "\n========================================" << endl;
	cout << "  GLOSSARIO DE PARAMETROS (IMM)" << endl;
	cout << "========================================" << endl;
	cout << "- ell     : Fator de confianca probabilistica." << endl;
	cout << "- eps'    : Margem de erro ajustada para a Fase 1." << endl;
	cout << "- theta_i : Meta dinamica de amostras necessarias nesta iteracao." << endl;
	cout << "- H       : Total absoluto de amostras (RR-Sets) geradas." << endl;
	cout << "- totW    : Volume estrutural do hipergrafo (total de arestas bipartidas)." << endl;
	cout << "- OPT_    : Estimativa do Limite Inferior do spread otimo." << endl;
	cout << "- theta   : Meta final e definitiva de amostras para a Fase 2." << endl;
	cout << "- Inf(S)  : Estimativa teorica imparcial do spread (nos impactados)." << endl;
	cout << "========================================\n" << endl;

	printf("ell  = %f\n", ell);
	printf("eps' = %f\n", eps_p);
	printf("log{V c k} = %f\n", log_VCk);

	double OPT_lb = 1; ///< Limite Inferior do Ótimo (Lower Bound of OPT)

	int H = 0; ///< Total atual de hiperarestas (RR-Sets) geradas
	
	/// h2v: Mapeia [ID do RR-Set] -> [Lista de Nós alcançados por este RR-Set]
	vector<vector<int> > h2v;
	
	/// v2h: Mapeia [ID do Nó] -> [Lista de RR-Sets que alcançam este nó]
	vector<vector<int> > v2h(V);
	
	long long int totW = 0; ///< Somatório da quantidade de vértices nos RR-sets gerados até então.

	// ========================================================================
	// FASE 1: ESTIMAÇÃO DO LIMITE INFERIOR (OPT Lower Bound)
	// ========================================================================
	/// O objetivo desta fase é encontrar uma estimativa robusta (OPT_lb) do spread máximo 
	/// possível sem gerar hiperarestas desnecessárias. Fazemos isso testando iterativamente 
	/// limiares logarítmicos.
	for (int i = 1; i <= log2(V) - 1; i++) {
		double x = V / pow(2, i);
		
		/// $ \lambda' $ é derivado dos limites de Chernoff. Ele define a densidade necessária
		/// para validar a qualidade da aproximação nesta iteração i.
		double lambda_prime = (2 + 2.0 / 3.0 * eps_p)
			* (log_VCk + ell * log(V) + log(log2(V))) * V / (eps_p * eps_p);
		
		/// $ \theta_i $ é a meta cumulativa de RR-Sets a serem gerados até esta iteração.
		double theta_i = lambda_prime / x;

		printf("i = %d\n", i);
		//printf("x  = %.0f\n", x);
		printf("theta_i = %.0f\n", theta_i);

		/// Calcula quantos RR-Sets faltam para atingir o limiar $ \theta_i $.
		long long iterations_needed = (long long)(theta_i - H);

		/// Gera os RR-Sets faltantes e atualiza as estruturas h2v e v2h paralelamente.
		generate_samples(V, rs, model, iterations_needed, h2v, v2h, totW, i);

		H = h2v.size(); // Atualiza H real após a inserção em bloco

		printf("H  = %d\n", H);
		printf("totW = %lld\n", totW);

		vector<int> S;
		/// Executa o Algoritmo Guloso sobre a amostra atual para encontrar o melhor conjunto provisório.
		/// A máscara 'candidates' é repassada para garantir a restrição de universo.
		int degS = greedy(V, h2v, v2h, k, S, candidates);
		
		/// deg(S) representa a quantidade de RR-Sets cobertos pelo conjunto S encontrado nesta iteração.
		/// Inf(S) é a estimativa de influência normalizada, calculada como $ V * deg(S) / H $.
		//printf("deg(S) = %d\n", degS);
		printf("Inf(S) = %f\n", 1.0 * V * degS / H);
		printf("\n");

		/// Condição de Parada Baseada no Martingale:
		/// Se a cobertura da semente provisória normalizada pelo limiar $ \theta_i $ exceder
		/// o critério de margem, inferimos que encontramos uma estimativa segura para o Limite Inferior (OPT_lb).
		if (1.0 * V * degS / theta_i >= (1 + eps_p) * x) {
			OPT_lb = (1.0 * V * degS) / ((1 + eps_p) * theta_i);
			break;
		}
	}

	// ========================================================================
	// CÁLCULO DE PARÂMETROS PARA A FASE DE REFINAMENTO
	// ========================================================================
	double lambda_star;
	{
		/// Cálculo rigoroso de $ \alpha $ e $ \beta $ baseado nas falhas probabilísticas de Chernoff.
		double alpha = sqrt(ell * log(V) + log(2));
		double beta = sqrt((1 - 1 / e) * (log_VCk + ell * log(V) + log(2)));
		double c = (1 - 1 / e) * alpha + beta;
		
		/// $ \lambda^* $ é o núcleo da restrição de aproximação do IMM.
		lambda_star = 2 * V * c * c / (eps * eps);
	}
	
	/// $ \theta $ representa o número TOTAL E DEFINITIVO de RR-Sets necessários.
	/// Quanto maior o OPT_lb encontrado na Fase 1, MENOS RR-sets precisaremos na Fase 2.
	double theta = lambda_star / OPT_lb;
	
	printf("OPT_ = %.0f\n", OPT_lb);
	//printf("lambda* = %.0f\n", lambda_star);
	printf("theta = %.0f\n", theta);

	// ========================================================================
	// FASE 2: REFINAMENTO (Geração Final e Busca Exata)
	// ========================================================================
	long long iterations_needed_2 = (long long)(theta - H);

	/// Completa o volume de amostras até o limite definitivo $ \theta $.
	/// O salt (67890) é injetado para assegurar que as cadeias pseudoaleatórias desta fase
	/// não se correlacionem indesejadamente com as cadeias da Fase 1, preservando a independência do Martingale.
	generate_samples(V, rs, model, iterations_needed_2, h2v, v2h, totW, 67890);

	H = h2v.size();
	printf("H  = %d\n", H);

	vector<int> S;
	/// Executa o Algoritmo Guloso (CELF) pela última vez, agora sobre o hipergrafo completo.
	/// O conjunto 'S' retornado aqui garante o limite teórico de $ 1 - 1/e - \epsilon $.
	int degS = greedy(V, h2v, v2h, k, S, candidates);
	
	printf("deg(S) = %d\n", degS);
	printf("Inf(S) = %f\n", 1.0 * V * degS / H);

	return S;
}