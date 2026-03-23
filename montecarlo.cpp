/**
 * @brief Simula a propagação de influência utilizando o modelo Independent Cascade (IC).
 * * @details Esta função executa uma simulação estocástica "para frente" (forward propagation).
 * Partindo de um conjunto inicial de sementes ativas, ela utiliza uma Busca em Largura (BFS) 
 * para propagar a ativação. No modelo IC, quando um nó 'u' é ativado no tempo t, ele tem 
 * uma única e exclusiva chance de ativar cada vizinho inativo 'v' no tempo t+1, com 
 * probabilidade dependente do peso da aresta (u -> v).
 * * @note Arquitetura Thread-Safe: A função aloca o vetor de estados `active` localmente. 
 * Como esta função é executada dentro de um bloco paralelo (OpenMP), o uso de uma estrutura
 * local (ao invés de limpar uma estrutura global) previne falhas de concorrência (Race Conditions) 
 * e o gargalo de 'false sharing' no cache da CPU. Para redes de dezenas de milhares de nós, 
 * o custo de alocação de um vector<bool> local é ínfimo.
 * * @param V Número total de vértices (nós) na rede. Utilizado para dimensionar estruturas e garantir limites.
 * @param es Grafo Direto (Edge Structure). es[u] contém as arestas que saem de 'u' e apontam para seus vizinhos.
 * @param S O conjunto inicial de sementes (os "pacientes zero" da difusão).
 * @param gen Referência para o motor Mersenne Twister da thread atual, garantindo amostragens independentes.
 * * @return int O número total e absoluto de nós que foram ativados (influenciados) ao final desta simulação.
 */
int MonteCarlo_IC(int V, vector<vector<edge> >& es, vector<int>& S, mt19937& gen) {

	queue<int> Q; ///< Fila de processamento para a propagação em onda (BFS).
	int count = 0; ///< Acumulador do número de nós ativados nesta rodada específica.
	
	/// Distribuição uniforme contínua para simular a "jogada de moeda" da propagação.
	uniform_real_distribution<> dis_prob(0.0, 1.0);

	// ========================================================================
	// 1. INICIALIZAÇÃO DE ESTADO
	// ========================================================================
	/// ALOCAÇÃO SEGURA (LOCAL)
	/// O vector<bool> é otimizado pelo C++ para usar apenas 1 bit por elemento.
	/// Para V=75.000 (ex: base segmentada de investidores), isso consome cerca de 9KB de RAM.
	/// O isolamento na stack da thread garante execução assíncrona limpa.
	vector<bool> active(V, false);

	/// Ativa o conjunto semente inicial e os enfileira para a primeira onda de propagação.
	for (int s : S) {
		// Proteção extra: Bound check implícito para evitar falhas de segmentação caso 
		// uma semente fora do domínio seja repassada acidentalmente.
		if (s < V && !active[s]) {
			active[s] = true;
			Q.push(s);
			count++;
		}
	}

	// ========================================================================
	// 2. LOOP DE PROPAGAÇÃO EM CASCATA (BFS)
	// ========================================================================
	while (!Q.empty()) {
		int u = Q.front();
		Q.pop();

		/// Itera sobre todas as arestas de saída do nó recém-ativado 'u'.
		for (auto& e : es[u]) {
			int v = e.v; // Nó de destino que pode receber a influência.

			// Proteção de limites e verificação de redundância (se 'v' já está ativo, 
			// ele não pode ser reativado, honrando a premissa de estado único do modelo IC).
			if (v < V && !active[v]) {
				
				/// Teste de Ativação (Roleta Russa):
				/// Se o número aleatório sorteado for menor ou igual à probabilidade da aresta,
				/// a propagação tem sucesso.
				if (dis_prob(gen) <= e.c) {
					active[v] = true; // Muda o estado do vizinho para ativo.
					Q.push(v);        // Coloca na fila para que, no próximo ciclo, ele tente influenciar sua própria rede.
					count++;          // Incrementa o impacto total da campanha.
				}
			}
		}
	}
	
	/// Retorna o "spread" final desta única simulação estocástica.
	return count;
}