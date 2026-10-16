#include <catch2/catch_test_macros.hpp>

#include "arena.hpp"
#include "tensor.hpp"

TEST_CASE("TensorShape basics", "[tensor]")
{
    TensorShape s1{2, 3, 4};
    CHECK(s1.ndim == 3);
    CHECK(s1.dims[0] == 2);
    CHECK(s1.dims[1] == 3);
    CHECK(s1.dims[2] == 4);
    CHECK(s1.Numel() == 24);

    TensorShape s2{2, 3, 4};
    CHECK(s1 == s2);

    TensorShape s3{2, 3, 5};
    CHECK(s1 != s3);
}

TEST_CASE("Tensor creation and standard strides", "[tensor]")
{
    Arena arena(KiB(64));
    Tensor<f32>* t = Tensor<f32>::Create(arena, {2, 3, 4});
    REQUIRE(t != nullptr);
    CHECK(t->ndim == 3);
    CHECK(t->shape[0] == 2);
    CHECK(t->shape[1] == 3);
    CHECK(t->shape[2] == 4);
    CHECK(t->strides[0] == 12);
    CHECK(t->strides[1] == 4);
    CHECK(t->strides[2] == 1);
    CHECK(t->Numel() == 24);
    CHECK(t->IsContiguous());
}

TEST_CASE("Tensor indexing 1D, 2D, 3D, 4D", "[tensor]")
{
    Arena arena(KiB(64));
    auto* t3 = Tensor<f32>::Create(arena, {2, 3, 4});
    REQUIRE(t3 != nullptr);

    f32 counter = 1.0f;
    for (u32 i = 0; i < 2; ++i) {
        for (u32 j = 0; j < 3; ++j) {
            for (u32 k = 0; k < 4; ++k) {
                t3->At(i, j, k) = counter++;
            }
        }
    }

    CHECK(t3->At(0, 0, 0) == 1.0f);
    CHECK(t3->At(1, 2, 3) == 24.0f);
    CHECK(t3->At(0, 1, 2) == 7.0f);

    // Span check
    auto s = t3->Span();
    CHECK(s.size() == 24);
    CHECK(s[0] == 1.0f);
    CHECK(s[23] == 24.0f);
}

TEST_CASE("Tensor Reshape", "[tensor]")
{
    Arena arena(KiB(64));
    auto* t = Tensor<f32>::Create(arena, {2, 6});
    for (u32 i = 0; i < 12; ++i) {
        t->data[i] = static_cast<f32>(i + 1);
    }

    auto r = t->Reshape({3, 4});
    REQUIRE(r.has_value());
    CHECK(r->ndim == 2);
    CHECK(r->shape[0] == 3);
    CHECK(r->shape[1] == 4);
    CHECK(r->strides[0] == 4);
    CHECK(r->strides[1] == 1);
    CHECK(r->At(0, 0) == 1.0f);
    CHECK(r->At(2, 3) == 12.0f);

    auto r3d = t->Reshape({2, 2, 3});
    REQUIRE(r3d.has_value());
    CHECK(r3d->ndim == 3);
    CHECK(r3d->At(1, 1, 2) == 12.0f);

    // Incompatible size fails
    CHECK_FALSE(t->Reshape({5, 5}).has_value());
}

TEST_CASE("Tensor Transpose and Permute (zero-copy)", "[tensor]")
{
    Arena arena(KiB(64));
    auto* t = Tensor<f32>::Create(arena, {2, 3});
    t->At(0, 0) = 1.0f; t->At(0, 1) = 2.0f; t->At(0, 2) = 3.0f;
    t->At(1, 0) = 4.0f; t->At(1, 1) = 5.0f; t->At(1, 2) = 6.0f;

    auto tr = t->Transpose(0, 1);
    REQUIRE(tr.has_value());
    CHECK(tr->ndim == 2);
    CHECK(tr->shape[0] == 3);
    CHECK(tr->shape[1] == 2);
    CHECK(tr->strides[0] == 1);
    CHECK(tr->strides[1] == 3);
    CHECK_FALSE(tr->IsContiguous());

    CHECK(tr->At(0, 0) == 1.0f);
    CHECK(tr->At(1, 0) == 2.0f);
    CHECK(tr->At(2, 0) == 3.0f);
    CHECK(tr->At(0, 1) == 4.0f);
    CHECK(tr->At(1, 1) == 5.0f);
    CHECK(tr->At(2, 1) == 6.0f);

    // Permute 3D
    auto* t3 = Tensor<f32>::Create(arena, {2, 3, 4});
    t3->Fill(0.0f);
    t3->At(1, 2, 3) = 42.0f;

    auto p = t3->Permute({2, 0, 1});
    REQUIRE(p.has_value());
    CHECK(p->ndim == 3);
    CHECK(p->shape[0] == 4);
    CHECK(p->shape[1] == 2);
    CHECK(p->shape[2] == 3);
    CHECK(p->At(3, 1, 2) == 42.0f);
}

TEST_CASE("Tensor Slice (zero-copy)", "[tensor]")
{
    Arena arena(KiB(64));
    auto* t = Tensor<f32>::Create(arena, {5, 6});
    for (u32 r = 0; r < 5; ++r) {
        for (u32 c = 0; c < 6; ++c) {
            t->At(r, c) = static_cast<f32>(r * 10 + c);
        }
    }

    // Slice rows [1, 4)
    auto s_row = t->Slice(0, 1, 4);
    REQUIRE(s_row.has_value());
    CHECK(s_row->shape[0] == 3);
    CHECK(s_row->shape[1] == 6);
    CHECK(s_row->At(0, 0) == 10.0f);
    CHECK(s_row->At(2, 5) == 35.0f);

    // Slice cols with step 2
    auto s_col = s_row->Slice(1, 1, 6, 2);
    REQUIRE(s_col.has_value());
    CHECK(s_col->shape[0] == 3);
    CHECK(s_col->shape[1] == 3); // cols 1, 3, 5
    CHECK(s_col->At(0, 0) == 11.0f);
    CHECK(s_col->At(0, 1) == 13.0f);
    CHECK(s_col->At(0, 2) == 15.0f);
    CHECK(s_col->At(1, 0) == 21.0f);
    CHECK(s_col->At(2, 2) == 35.0f);
}

TEST_CASE("Tensor BroadcastTo (NumPy semantics)", "[tensor]")
{
    Arena arena(KiB(64));

    SECTION("1D [3] to 2D [4, 3]")
    {
        auto* t = Tensor<f32>::Create(arena, {3});
        t->At(0) = 10.0f;
        t->At(1) = 20.0f;
        t->At(2) = 30.0f;

        auto b = t->BroadcastTo({4, 3});
        REQUIRE(b.has_value());
        CHECK(b->ndim == 2);
        CHECK(b->shape[0] == 4);
        CHECK(b->shape[1] == 3);
        CHECK(b->strides[0] == 0);
        CHECK(b->strides[1] == 1);
        CHECK_FALSE(b->IsContiguous());

        for (u32 r = 0; r < 4; ++r) {
            CHECK(b->At(r, 0) == 10.0f);
            CHECK(b->At(r, 1) == 20.0f);
            CHECK(b->At(r, 2) == 30.0f);
        }
    }

    SECTION("2D [4, 1] to 2D [4, 3]")
    {
        auto* t = Tensor<f32>::Create(arena, {4, 1});
        t->At(0, 0) = 1.0f;
        t->At(1, 0) = 2.0f;
        t->At(2, 0) = 3.0f;
        t->At(3, 0) = 4.0f;

        auto b = t->BroadcastTo({4, 3});
        REQUIRE(b.has_value());
        CHECK(b->shape[0] == 4);
        CHECK(b->shape[1] == 3);
        CHECK(b->strides[0] == 1);
        CHECK(b->strides[1] == 0);

        for (u32 r = 0; r < 4; ++r) {
            for (u32 c = 0; c < 3; ++c) {
                CHECK(b->At(r, c) == static_cast<f32>(r + 1));
            }
        }
    }

    SECTION("BroadcastShapes helper")
    {
        u32 out[kMaxTensorDims] = {};
        u32 out_ndim = 0;
        const u32 sh1[] = {1, 3};
        const u32 sh2[] = {4, 1};
        REQUIRE(BroadcastShapes(sh1, sh2, out, out_ndim));
        CHECK(out_ndim == 2);
        CHECK(out[0] == 4);
        CHECK(out[1] == 3);

        const u32 bad1[] = {2, 3};
        const u32 bad2[] = {4, 3};
        CHECK_FALSE(BroadcastShapes(bad1, bad2, out, out_ndim));
    }
}

TEST_CASE("Tensor Contiguous copy from non-contiguous view", "[tensor]")
{
    Arena arena(KiB(64));
    auto* t = Tensor<f32>::Create(arena, {2, 3});
    t->At(0, 0) = 1; t->At(0, 1) = 2; t->At(0, 2) = 3;
    t->At(1, 0) = 4; t->At(1, 1) = 5; t->At(1, 2) = 6;

    auto tr = t->Transpose(0, 1);
    REQUIRE(tr.has_value());
    CHECK_FALSE(tr->IsContiguous());

    auto* c = tr->Contiguous(arena);
    REQUIRE(c != nullptr);
    CHECK(c->IsContiguous());
    CHECK(c->shape[0] == 3);
    CHECK(c->shape[1] == 2);
    CHECK(c->strides[0] == 2);
    CHECK(c->strides[1] == 1);

    auto span = c->Span();
    CHECK(span[0] == 1);
    CHECK(span[1] == 4);
    CHECK(span[2] == 2);
    CHECK(span[3] == 5);
    CHECK(span[4] == 3);
    CHECK(span[5] == 6);
}

TEST_CASE("Tensor Mat interop", "[tensor]")
{
    Arena arena(KiB(64));
    Mat* m = Mat::Create(arena, 2, 3);
    m->At(0, 0) = 10.0f;
    m->At(1, 2) = 20.0f;

    Tensor<f32> t = Tensor<f32>::FromMat(*m);
    CHECK(t.ndim == 2);
    CHECK(t.shape[0] == 2);
    CHECK(t.shape[1] == 3);
    CHECK(t.At(0, 0) == 10.0f);
    CHECK(t.At(1, 2) == 20.0f);

    Mat m_back = t.ToMat();
    CHECK(m_back.rows == 2);
    CHECK(m_back.cols == 3);
    CHECK(m_back.At(0, 0) == 10.0f);
    CHECK(m_back.At(1, 2) == 20.0f);
}

TEST_CASE("OwnedTensor heap management", "[tensor]")
{
    OwnedTensor<f32> ot({3, 4});
    ot.View().At(1, 2) = 99.0f;

    Tensor<f32>& view = ot;
    CHECK(view.At(1, 2) == 99.0f);
    CHECK(view.IsContiguous());
    CHECK(view.Numel() == 12);
}
