/**
 * @brief Algoritmo Guloso otimizado com CELF (Cost-Effective Lazy Forward) para seleção de sementes.
 * * @details Esta função resolve o problema de cobertura máxima (Max-Cover) sobre o hipergrafo
 * bipartido de Conjuntos de Alcance Reverso (RR-Sets). Devido à propriedade de submodularidade
 * da influência, o ganho marginal de um nó é monotonicamente decrescente. O CELF tira proveito
 * disso utilizando uma Fila de Prioridade com avaliação "preguiçosa" (lazy evaluation),
 * recalculando graus apenas quando estritamente necessário.
 * * @param V Número total de vértices na rede original.
 * @param h2v Mapeamento de hiperarestas para vértices (Quais nós pertencem ao RR-Set H_i).
 * @param v2h Mapeamento de vértices para hiperarestas (Índice invertido: Quais RR-Sets o nó v cobre).
 * @param k O limite orçamentário (budget), ou seja, o número exato de sementes a serem selecionadas.
 * @param S Vetor passado por referência que será populado com os IDs das k sementes selecionadas.
 * @param candidates Máscara de filtro de universo. Se candidates[v] == false, o nó é sumariamente
 * ignorado durante a inicialização.
 * * @return int O grau total (degS), representando a quantidade absoluta de hiperarestas exclusivas
 * cobertas pelo conjunto final de sementes S.
 */
int greedy(int V, vector<vector<int> >& h2v, vector<vector<int> >& v2h, int k,
	vector<int>& S, const vector<bool>& candidates) { // <--- NOVO PARAMETRO

	int H = (int)h2v.size();

	/// Vetor de estado das hiperarestas. Se dead[h] == true, este RR-Set já foi "capturado" 
	/// por uma semente previamente selecionada e não oferece mais ganho marginal.
	vector<bool> dead(H, false);

	/// Armazena o grau marginal atualizado de cada vértice.
	vector<int> deg(V);

	/// Fila de Prioridade Máxima (Max-Heap) para a mecânica CELF.
	/// Armazena pares <grau_marginal_armazenado, id_do_no>. Mantém o nó mais promissor no topo.
	priority_queue<pair<int, int> > Q;

	// ========================================================================
	// 1. INICIALIZAÇÃO E FILTRO DE UNIVERSO
	// ========================================================================
	for (int v = 0; v < V; v++) {

		// --- RESTRIÇÃO DE NEGÓCIO ---
		// Se o nó não faz parte do universo admissível estipulado pela área de negócios 
		// (ex: clientes sem saldo mínimo), não o inserimos na fila, impossibilitando sua seleção.
		if (!candidates[v]) continue;

		/// O grau inicial de um nó é simplesmente o número de RR-Sets que ele alcança.
		deg[v] = (int)v2h[v].size();

		/// Ignora nós isolados que não alcançaram nenhum cenário de simulação.
		if (deg[v] > 0) {
			Q.push(make_pair(deg[v], v));
		}
	}

	int total_covered = 0; ///< Acumulador absoluto de RR-Sets cobertos por S.
	vector<bool> selected(V, false); ///< Rastreador de sementes para evitar seleções duplicadas.

	// ========================================================================
	// 2. LOOP LAZY EVALUATION (O Coração do CELF)
	// ========================================================================
	/// Executa até preenchermos o orçamento (k) ou exaurirmos as opções viáveis.
	while (S.size() < k && !Q.empty()) {

		/// Remove e analisa o candidato "aparentemente" melhor.
		pair<int, int> top = Q.top();
		Q.pop();

		int v = top.second;
		int stored_deg = top.first; // O ganho marginal na última vez que este nó foi avaliado.

		if (selected[v]) continue; // Prevenção de redundância.

		/// A MÁGICA DA SUBMODULARIDADE:
		/// Se o ganho marginal que estava guardado na fila (stored_deg) ainda é igual ao 
		/// ganho marginal real/atual deste nó (deg[v]), temos certeza matemática de que 
		/// NENHUM outro nó na fila pode ser melhor do que ele neste momento.
		if (stored_deg == deg[v]) {

			/// 1. Seleciona o nó como semente definitiva.
			S.push_back(v);
			selected[v] = true;
			total_covered += deg[v];

			/// 2. Atualização em cascata (Descontando a interseção)
			/// Iteramos por todos os RR-Sets que este nó 'v' recém-adicionado alcança.
			for (int h_idx : v2h[v]) {

				/// Se o RR-Set ainda estava "vivo" (não coberto):
				if (!dead[h_idx]) {
					dead[h_idx] = true; // Agora ele está coberto (morto).

					/// Para manter a contabilidade correta, subtraímos 1 do ganho marginal de TODOS 
					/// os outros nós que também alcançavam este exato RR-Set, pois este hipergrafo 
					/// não fornece mais ganho marginal a ninguém.
					for (int u : h2v[h_idx]) {
						deg[u]--;
					}
				}
			}
		}
		else {
			/// Se stored_deg != deg[v], significa que o nó 'v' perdeu potencial desde a sua última
			/// avaliação (porque alguma semente escolhida anteriormente cobriu RR-Sets que ele cobria).
			/// Nós simplesmente atualizamos seu valor na fila e o inserimos novamente.
			Q.push(make_pair(deg[v], v));
		}
	}

	return total_covered;
}