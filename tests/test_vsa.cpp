#include "core/vsa.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace cognicore;

void test_orthogonality() {
    auto a = HyperVector::random(101);
    auto b = HyperVector::random(202);
    auto c = HyperVector::from_token("fogo");
    auto d = HyperVector::from_token("agua");

    assert(a.similarity(a) == 1.0);
    assert(std::abs(a.similarity(b)) < 0.08);
    assert(std::abs(c.similarity(d)) < 0.08);
    std::cout << "[PASS] test_orthogonality (sim(a,b)=" << a.similarity(b) << ")\n";
}

void test_bind_and_permute() {
    auto role = HyperVector::from_token("SUJEITO");
    auto val = HyperVector::from_token("CALOR");
    auto bound = role.bind(val);

    // Quasi-orthogonal to both operands
    assert(std::abs(bound.similarity(role)) < 0.08);
    assert(std::abs(bound.similarity(val)) < 0.08);

    // Exact unbind
    assert(bound.unbind(role) == val);
    assert(bound.unbind(val) == role);

    // Permute invertibility and wrap-around
    assert(val.permute(0) == val);
    assert(val.permute(static_cast<int>(VSA_BITS)) == val);
    assert(val.permute(1) != val);
    assert(val.permute(17).permute(-17) == val);
    assert(val.permute(137).permute(-137) == val);
    assert(std::abs(val.similarity(val.permute(1))) < 0.08);

    std::cout << "[PASS] test_bind_and_permute\n";
}

void test_bundle_recovery() {
    auto r_subj = HyperVector::from_token("ROLE_SUBJ");
    auto r_rel = HyperVector::from_token("ROLE_REL");
    auto r_obj = HyperVector::from_token("ROLE_OBJ");

    auto v_fogo = HyperVector::from_token("fogo");
    auto v_causa = HyperVector::from_token("causa");
    auto v_calor = HyperVector::from_token("calor");
    auto v_gelo = HyperVector::from_token("gelo");

    auto record = HyperVector::bundle({
        r_subj.bind(v_fogo),
        r_rel.bind(v_causa),
        r_obj.bind(v_calor)
    });

    auto recovered_obj = record.unbind(r_obj);
    double sim_calor = recovered_obj.similarity(v_calor);
    double sim_gelo = recovered_obj.similarity(v_gelo);
    double sim_fogo = recovered_obj.similarity(v_fogo);

    assert(sim_calor > 0.35);
    assert(std::abs(sim_gelo) < 0.08);
    assert(std::abs(sim_fogo) < 0.08);

    std::cout << "[PASS] test_bundle_recovery (sim_calor=" << sim_calor
              << ", sim_gelo=" << sim_gelo << ")\n";
}

void test_random_indexing_codebook() {
    Codebook cb;
    // "fogo" and "chama" share identical contexts; "peixe" has a disjoint context
    for (int i = 0; i < 5; ++i) {
        cb.observe_context("fogo", {"calor", "queima", "luz", "fumaca"});
        cb.observe_context("chama", {"calor", "queima", "luz", "fumaca"});
        cb.observe_context("peixe", {"agua", "oceano", "nadadeira", "mar"});
    }
    cb.finalize_semantics();

    auto id_fogo = cb.find_id("fogo");
    auto id_chama = cb.find_id("chama");
    auto id_peixe = cb.find_id("peixe");

    double sim_same = cb.semantic_vector(id_fogo).similarity(cb.semantic_vector(id_chama));
    double sim_diff = cb.semantic_vector(id_fogo).similarity(cb.semantic_vector(id_peixe));

    assert(sim_same > 0.55);
    assert(std::abs(sim_diff) < 0.15);

    auto nearest = cb.query_nearest(cb.semantic_vector(id_fogo), 3, true);
    assert(nearest.size() == 3);
    assert(nearest[0].first == "fogo");
    assert(nearest[1].first == "chama");

    std::cout << "[PASS] test_random_indexing_codebook (sim(fogo,chama)=" << sim_same
              << ", sim(fogo,peixe)=" << sim_diff << ")\n";
}

void test_loaded_semantics_survive_incremental_ingestion() {
    Codebook trained;
    trained.observe_context("leitura", {"livro", "texto", "ideias"});
    trained.finalize_semantics();
    const auto expected = trained.semantic_vector(trained.find_id("leitura"));
    Codebook loaded;
    loaded.load_raw(trained.tokens(), trained.base_vectors(), trained.semantic_vectors());
    loaded.observe_context("conversa", {"amizade", "diálogo"});
    loaded.finalize_semantics();
    assert(loaded.semantic_vector(loaded.find_id("leitura")) == expected);
    assert(loaded.semantic_vector(loaded.find_id("conversa")) != loaded.base_vector(loaded.find_id("conversa")));
}

int main() {
    test_orthogonality();
    test_bind_and_permute();
    test_bundle_recovery();
    test_random_indexing_codebook();
    test_loaded_semantics_survive_incremental_ingestion();
    return 0;
}
