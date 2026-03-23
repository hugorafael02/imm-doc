/**
 * @brief Gera um único Reverse Reachable Set (RR-Set) sob o modelo Independent Cascade (IC).
 * * @details Esta função simula a propagação reversa de influência. Ela seleciona um nó raiz
 * uniformemente ao acaso e executa uma Busca em Largura (BFS) estocástica utilizando a
 * topologia transposta do grafo. Uma aresta (v -> u) é "ativada" se um número aleatório
 * sorteado for menor ou igual à probabilidade de propagação da aresta.
 * * @note Otimização de Memória: Em vez de alocar um vetor booleano de tamanho V (o que
 * consumiria memória excessiva e causaria overhead de inicialização a cada chamada),
 * a função utiliza um `std::unordered_set` (Tabela Hash) para rastrear os nós visitados,
 * garantindo tempo de busca médio O(1) e alocação estritamente proporcional ao tamanho do RR-Set.
 * * @param V Número total de vértices na rede. Define o limite superior para o sorteio do nó raiz.
 * @param rs Grafo transposto (Reverse Structure). rs[u] contém todas as arestas originais (v -> u)
 * indicando os nós 'v' que exercem influência sobre 'u'.
 * @param gen Referência para o motor de números pseudoaleatórios (Mersenne Twister) da thread atual.
 * * @return std::vector<int> Lista contendo os IDs de todos os nós que alcançaram o nó raiz
 * nesta simulação específica. Este é o chamado "RR-Set".
 */
vector<int> gen_RR_IC(int V, vector<vector<edge> >& rs, mt19937& gen) {

	/// Distribuição uniforme discreta para sortear o nó raiz (alvo da influência) no intervalo [0, V-1].
	uniform_int_distribution<> dis_node(0, V - 1);

	/// Distribuição uniforme contínua para atuar como a "moeda" da propagação no intervalo [0.0, 1.0].
	uniform_real_distribution<> dis_prob(0.0, 1.0);

	vector<int> RR; ///< Armazena o conjunto final de nós que formam o RR-Set.
	queue<int> Q;   ///< Fila padrão para o processamento da Busca em Largura (BFS).

	// ========================================================================
	// CONTROLE DE CICLOS E VISITAS (Otimização Hash)
	// ========================================================================
	/// USAMOS UM SET PARA NÃO ALOCAR VETOR GIGANTE A CADA ITERAÇÃO.
	/// Um vetor bool(V, false) alocaria dinamicamente bytes na RAM para toda a rede a cada 
	/// amostragem. Em grafos grandes, isso destrói a performance do heap e polui o cache L1/L2.
	/// O unordered_set aloca memória apenas para os nós efetivamente descobertos.
	unordered_set<int> visited;

	/// 1. Seleção da Raiz (O "Paciente Zero" reverso)
	int z = dis_node(gen);

	visited.insert(z); // Marca o nó raiz como visitado para evitar auto-loops.
	RR.push_back(z);
	Q.push(z);

	// ========================================================================
	// BUSCA EM LARGURA ESTOCÁSTICA (Reverse BFS)
	// ========================================================================
	while (!Q.empty()) {
		/// Retira o primeiro nó da fila de processamento.
		int u = Q.front();
		Q.pop();

		/// Itera sobre todos os "vizinhos de entrada" (nós que influenciam 'u').
		for (auto& e : rs[u]) {

			/// e.u representa o nó de ORIGEM no grafo direto, que agora é o nosso destino
			/// na caminhada reversa. (v -> u no original, u -> v na busca).
			int v = e.u;

			/// Verifica se o nó 'v' já foi integrado ao RR-Set nesta simulação.
			/// O método .count() de um unordered_set opera em tempo constante O(1).
			if (visited.count(v) == 0) {

				/// Roleta Russa da Influência:
				/// Sorteia um valor [0, 1]. Se for menor ou igual ao peso da aresta (e.c), a influência ocorreu.
				if (dis_prob(gen) <= e.c) {
					visited.insert(v);    // Marca como visitado para blindar contra ciclos fechados.
					RR.push_back(v);      // Registra o nó no RR-Set final.
					Q.push(v);            // Enfileira o nó para que seus próprios influenciadores sejam avaliados.
				}
			}
		}
	}

	return RR;
}