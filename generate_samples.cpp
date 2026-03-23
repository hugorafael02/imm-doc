/**
 * @brief Orquestra a geração massiva e paralela de Conjuntos de Alcance Reverso (RR-Sets).
 * * Esta função é responsável por amostrar cenários de propagação na rede transposta e
 * indexá-los no hipergrafo bipartido. Ela utiliza OpenMP para paralelizar a simulação estocástica
 * e aplica semântica de movimentação (move semantics) do C++11 para mitigar gargalos de I/O em memória RAM.
 * * @param V O número total de vértices (nós) na rede.
 * @param rs O grafo em sua forma transposta (Reverse Structure). Necessário para a busca reversa.
 * @param model O modelo de difusão a ser utilizado ("ic" para Independent Cascade, "lt" para Linear Threshold).
 * @param needed A cota exata de amostras (RR-sets) que precisam ser geradas nesta chamada.
 * @param h2v Estrutura do hipergrafo bipartido (Hiperaresta -> Vértices). Modificada in-place.
 * @param v2h Estrutura do hipergrafo bipartido (Vértice -> Hiperarestas). Modificada in-place para buscas rápidas no CELF.
 * @param totW O volume estrutural do hipergrafo (Total Width), representando o total de arestas neste grafo bipartido. Modificado in-place.
 * @param seed_salt Um modificador (salt) injetado no motor pseudoaleatório. Vital para garantir a independência
 * matemática dos Martingales entre as diferentes fases do algoritmo IMM.
 */
void generate_samples(int V, vector<vector<edge> >& rs, string model, long long needed,
	vector<vector<int> >& h2v, vector<vector<int> >& v2h,
	long long& totW, int seed_salt) {

	/// Prevenção contra chamadas vazias (comum nas iterações de convergência da Fase 1 do IMM).
	if (needed <= 0) return;

	/// Captura o tamanho atual da base de dados h2v. 
	/// Como esta função é chamada iterativamente, precisamos saber de onde recomeçar a indexação sequencial.
	int initial_H_size = h2v.size();

	// ========================================================================
	// 1. GERAÇÃO PARALELA DE RR-SETS (Map Phase)
	// ========================================================================
	/// Abre o bloco de paralelismo do OpenMP. A partir deste ponto, o código é executado
	/// simultaneamente pelas threads disponíveis no sistema.
#pragma omp parallel
	{
		/// Buffer local da thread. Em vez de todas as threads tentarem dar 'push_back'
		/// no vetor global h2v (o que exigiria locks lentos), cada thread acumula seus RR-Sets privadamente.
		vector<vector<int> > local_h2v;

		/// Inicialização Thread-Safe do motor Mersenne Twister.
		/// O uso de XOR (^) combinando o relógio, o ID da thread e o seed_salt garante que
		/// nenhuma thread gere a mesma cadeia de Random Walks que outra.
		unsigned long seed = (unsigned long)(time(NULL) ^ (omp_get_thread_num() * 12345 + seed_salt));
		mt19937 gen(seed);

		/// Distribui a carga de trabalho 'needed' entre as threads.
		/// O 'nowait' indica que, assim que uma thread terminar sua cota, ela NÃO precisa
		/// esperar pelas outras para prosseguir para o bloco de agregação (critical section).
#pragma omp for nowait
		for (long long j = 0; j < needed; j++) {
			vector<int> RR;

			/// Seleciona o gerador de hiperaresta baseado no modelo dinâmico.
			if (model == "tvlt" || model == "lt") {
				RR = gen_RR_LT(V, rs, gen); /// Caminhada Aleatória (Random Walk) Linear
			}
			else {
				// Fallback padrão para o Independent Cascade
				RR = gen_RR_IC(V, rs, gen); /// Busca em Largura (BFS) Estocástica
			}

			/// Armazena no buffer local da thread (rápido, sem contenção de concorrência).
			local_h2v.push_back(RR);
		}

		// ========================================================================
		// AGREGAÇÃO GLOBAL (Reduce Phase)
		// ========================================================================
		/// Seção Crítica: Apenas uma thread por vez pode executar este bloco.
		/// É o gargalo necessário para unificar os resultados em 'h2v'.
#pragma omp critical
		{
			for (auto& rr : local_h2v) {
				/// OTIMIZAÇÃO CRÍTICA: std::move 
				/// Ao invés de copiar cada elemento do vetor (O(N) por hiperaresta), 
				/// nós roubamos o ponteiro de memória do vetor local e o transferimos 
				/// para o vetor global. Isso transforma uma operação de cópia pesada em uma operação O(1).
				h2v.push_back(std::move(rr));
			}
		}
	} // Fim da região paralela. Aqui ocorre uma barreira de sincronização implícita.

	// ========================================================================
	// 2. INDEXAÇÃO SEQUENCIAL (Atualização de v2h e totW)
	// ========================================================================
	/// Após todas as threads finalizarem a geração, precisamos construir o índice invertido (v2h).
	/// Este índice é essencial para que o algoritmo guloso (CELF) encontre rapidamente 
	/// quais hiperarestas um nó específico consegue alcançar.
	int final_H_size = h2v.size();

	/// Iteramos APENAS sobre as amostras recém-geradas nesta chamada (do initial ao final).
	for (int idx = initial_H_size; idx < final_H_size; idx++) {

		/// Para cada vértice (nó) capturado pela hiperaresta (RR-Set) atual:
		for (int v : h2v[idx]) {

			/// Adiciona a hiperaresta na lista de influência deste nó.
			v2h[v].push_back(idx);

			/// Incrementa o peso total do hipergrafo estrutural. 
			/// Esta métrica define diretamente o custo de tempo do algoritmo guloso subjacente.
			totW++;
		}
	}
}