# 【emmmbedding】不用存储的图床！

大家平时会用到图床服务吗？

部署图床服务需要很大的存储空间，而且云服务卖的硬盘通常是按容量×时间算钱的，所以做一个图床实际上要随时间花费O(n**2)的钱。

那有没有什么既能永久存储图片，又不用花钱的办法呢？

正好聪明的莉沫酱发明了不用存储的图床，有了它就不需要硬盘啦！


## 原理

其实是用了stable-diffusion的vae，用户上传了图片之后，就把图片压缩成一个很小的矩阵，用base64整个塞到url里，然后有人访问这个url的时候，再从url里还原出图片！

举个例子——

输入1张 316×316 的图片: 

![Lenna.jpg](example/Lenna.jpg)

经过vae编码之后，得到这个tensor，尺寸是`[1, 4, 39, 39]`:

```python
tensor([[[[ 1.0445e+01,  8.4394e+00,  6.9553e+00,  ...,  5.4966e+00,
            5.2353e+00,  8.0868e+00],
          [ 9.9430e+00,  6.4219e+00,  8.6987e+00,  ...,  6.0790e+00,
            8.8426e+00, -7.1120e-01],
          [ 8.4888e+00,  1.0169e+01,  1.0712e+01,  ...,  1.0351e+01,
            7.3863e-01,  2.0532e+00],
          ...,
```

然后把tensor经过uint8量化、webp压缩、base64，就可以得到一个url:

```
http://localhost:8000/image?q=CABFwUl9iUEnAAAAJwAAAFJJRkaIDwAAV0VCUFZQOCB8DwAAEDUAnQEqJwCcAD4VCINBIQYjn9sEAFEtIAKPH6i_u343eZv4z8n_b_yv9Ln-A77fIv-h9BP419g_x_9x_c32L_yXgj8E_5X1Avxz-V_4f8wv7X73nwHaJZd_PP-D_b_YC9UPmf-o_wv7rf5v0HP6D0A_H_6n_vfyx-gD-Q_yj_Jfmj7n_2z_ieKn9C_rP-l_rv5GfYB_E_55_mf7X_k_-9_p_o8_Yf9t_hv3e_03sm_K_6X_sf7f_jP_T_ofsB_jv80_yH9s_x__e_xP___8_3I-qL9T_Ym_S77x_3__-Z7Tk_Rnw8SDxDUR_RYD8_X_8isWuXaDTzYOZ38hA9NRlSH4ZviKIN78ZjqTCkXOFD6u46a84ugEpiGtyyt_4J0HPAKW7Cu6r2gquu2BwK0zTZN4gE3e4jg5Bm2HnfGNGBEMDhAf1ovjTK69YQIVMEPHRnBcl5NtrYiAuiFWQqTmRLuiBFFUTfjKvhU5cmxWHImccvHrv5C86E-mIWpuzHs8UnBzyqB7UtldoD6dqmdZaP7IlAJxXbrXTEEcQsEO6k3bJU_gFVznJ5fyNr8Z2Jf2AAD-_1BtfiNuVC4-xXQTeAqQsv40-dTe672HyAEuCN76jjTg9W_YZFQWSxG5uzmMKJCtNFWM3AWyVk3yKveZQqkoyZtK5uGGmXyZeO2kyDdDOLhPFDyS1tD-ppdQWJG8dAalb5_0ke0bHyQ31EZ_B7MuIC02GDLGvLWDUevP9iRUdwSERyedFCPvzYi5PYxyPrJesbdma8qIP0-48pKb2FMJbDeRh9B_TyrZMG6vuUDIDEijrg7-0Ok2X8Crnkywx2TsdvQFbZqtZx3d527crRSnjLIiYTqFtgBChYA0c-XEHJO9p7LF_-g-_WFJ1GPT54Emk_axs72QzQxDXKR80yL64GK6DRYhp0KbfAHw-d96p1qsFKEHv2InZBRAgkb2aURvlb-L6JRnKlpKlM6nnnJAAWNiFFy4WIrez4xDwEwvoMkq8GMBs3Qu1cbDAyZEDJy8nXp7_ruInbz91TgcuqawAEYVY3zDv0sVfoyZN9fZLtRJA5awIU3SmMrqyG8ZA7z_602eMW-bnuA63MLIn_uh1zIzRct2LalSIuggy17h7EPjrM42y3Nw5t7yxtfdygw7F-OpvYdP56cWjBXkfkqlnbB2xCvZGrNQSD9b9c4rux2JbGxlYhQfbqHagpj0dxEBxxsFqnmz-J1gixpT-mhTBRYhQm5kWV-2DB9ACkw6GvpNREheI2mnODfeAAm3MbtccUjmXRNLMhTcehjz4pP53PIgefZQmkTtzyP5ROGNrOmjieh4IX3CYH_m_mtpuy8ORF27SjyQbFwar4qxI3M-if2dl0_Mzj-HCnpbxmk7ZM0N0CjG5MhInzW8F4FAN9AEGmqkSU6dQ-JvPjXWKFmuzYNgsMBU7J5bwfHZWrhgLJWWFUfS6cLaFItSadFEppR76fiKGKR_3HtsuFPUfhzG5m9gkywhU7v10nnujv2QhopWHOvQ2w0X6wfHG--lzvGuuBPCX0IA95EanxrNjrV_N6r0fEfM-niD1_gxJb-hHxDxnUFiH1v4H41YnMwdLidchFq-JlcspDDAzBQ7TJjIQupCiEAdurar5AQ7GKbxjnmeMFkU2mgjsR6y9dMwm3FXkyeunhS2dT0RvG4h0U_ptceh_1VNdm2I8M8XyruQstJRTA1sql0Cw2QM-8QlCjhMK-4avb02OuNuis8X6n9dHz4hi9GVDpPeHYZxO6fcUiA7e61w2tOjfgI3Oar5tMH6ILX4nILy8CMW5slXYJwkp4pCJerFDdvbXIumSH0H8zVDwPm5uPYyjSYLp7eT_EzKwkLxNjTDjc2cZJnQrq5JEQMyhowVSzyNjKjYkmp2YvVkluMPBv3Uzd-URFQGWX1XzzZoxT6DECMPzZ8WOJ7ouLgr6ii1YiWADGZHdKBF7FAvRSOWu7IFVC4BKrjZmnIAvP8VBQKD6iQehCflrSks3bNxAmCpQZ6P32o1sRdH-S3IhIdnZWRzC92GUlI4bH8MFgp59_ujWrnlb7chKHS09j19Wsj9HG-htr0RqsU36Fc7s0qVMHcT_cCP8EzNVMEwctVkOlMjGXtWQ3fTep4bXsj72o9BRCEnEqP8VfAw34t-l_GyY1FUVPbXIbT8orVSKhU5dsj6TKZBXTAjh8TfB2-OxsS_j7BV-v5Hq3xUXu69GazqEyWlFHmWgDey21fSr7fhDzd6YzF7214pvfXt_exOZv_HggRFmKcrMuXyUKi9Y-lQfb0_K_yq-JwbOqB8NctbfYWEjXUkrbtJkvJimfMmI4jeEzEeDfRcXwd5ncxHkIVrfdQPfTS_hAfT9kNDmJJDN9Bj5WnsN9_hsMmUa0Wz2yV7jvhIKapNtiusNXHET3P4EAxqWaEXy4zkZK5FmkykCoBHouZDXRieUPAgCZtiBsoTbLKu8IK9FvCPnjs67EOfX-_2-Zx_PFlNY5tbusKDUnU9UEdLmQLA9b16gFJM7GrviDu2GKfWV0oHTzLeOuItz9KHaBagQzTewPcCoc1TBIQC-6CMVncQuHQKyiaAEyHq8qywZnQ3i94VeYinEkkjgJtq-9eNZH0xZoxqLbgc2YW4Oz4T2tkJkgWaQ6eSU0X8Bf-nNiHR-SElWuPi3KnvAP6i9OdkfcHoX3DI5bF2kl8RjQ9u4fPjcLV3p3VDE_wCRCYvMRqrUE_R3yxdPZwzc5zVg9HqgusMtKUeUIgnw3o3f6ne6h1q_Vq5AVJrH4L28lCQJfM_yAjnikGwhR08adMZW9jceg4JMMbqGspSu8gAV1y46sa-UbrxIRKdn5qhu_jpQct_FzqDyZgtzvGIi4IXTYJAe3xiV8rj0wnMxoj3mKpXrYvHPb38Ago6wRbLmblmxCKyY-534Ti4T1eZkMUY9Ktv0OjorXPuKFKZV2A_rii9UR5eQTq-TDRyaPQ3WAKgREqfDQFiJEB4sCye9S6Wwse6Zlf9-UP9HXOx7FsjVxytAtgjKPC0ANZsOgwU0AlvFyIboMUTk7I8eQ6tjJeK_zO1dZICb6i-HVhzC_Wv6HQzCxG-gbBncAZTaGQnZx3kMaYHE6QXhBVpaNHseQPGsOi19BqCIllujTdDy2PqEAuferRgY0a7mJy5n0V2ebAiJm6d4OHgNlJqAydiEmBXIWKYUou1gdWlTF98sYnd5DQz_ladnrLOi3DQhqFF94op2HbudnvqQP6wS04wxWn8HT_VEC2_Enz5M3rTREZ3kTpgohKWbCyxlWwMeR42fgJJeX2uq8WfgXnoZyLPYZg3JlzpEnSgRh0X4IB6P2DZ4T6EJisbIKZZAyMO4K1S2-NSxN0tpxlegRcgSdfKx5K71f8cf-QA6qGAfrMlyrzow0kAv4ogjttOxcYEAEQ1RRXS67RMboOM_8XRh-fNF0iL0MTOADQ_wASP8ZJFxi3KnmAIxAS2GbSdABcQZ4eujaVb99l00VcegKJFESh7CB_Kuv9aTYx_StIzRUpvTa5TJtZ5TrI7T-WumBKKeucnNKgHKYUjuvyqbAniuYZPjZY2qKstmjJxJ9ZzSd1xgp5UvJiUAelEuy5aJ-utXDz2gfu4bVO3gAZq3ZTb-J7X6zakxDNIiwvnaXFYSiocFXHokmVb35cNd0FZpj32DM9FdCTbgEPprGF030Xg7V6UZ_OtuGDCBJV0N_9aC74O2wGT78dyDEOdeXhElSvU3tdKFJ9I96LQnA8NPt7YNENuaFBmBp8dwueqOI9q8ChS2mZepyx1FfBgLlQkSsUsUoJx_MV02QsDp6-AQCky0Wrs1WPMQpBQuRu9FL6Xw-AroZ_X8z-UCboY3sPtXM93mYHGAucXKpuu50kndv8cMqZjLKbIOGHYAnHN6RJKvDL9a1XFUqb7aM1eLtKQ0GyzpgvTfOcCbAhOEpIdl_4M_qIFllSfcEDtj9VetAOUrRGSq5nPxtxeq_4bsKKr_BrtWzmkUbUqWGEii_gyNCcKV-fukYe-pnn8eTSBUMMsj8oriBoS2YX9_HNwNQ8_4-AIUPVOTiC30qzQ_qnAbZwwme_9xr-mIJRSyn1b2fRHY1EKbRp3Br2AUEaCNsvmxiPxCWFVmk_izD3fcIyRAByQWRq7iJuchs7MWX26MkGaMytn-dGg3h6OzUza_7UkuaPSwGzHAs-ruY_44tBp9lXfhp95UIQQ7Y5myisDnJHdKN96ChS6IUIoJMslihNz2C_gJM1dmF1u7qLVev4LgyROpGdhRdu9INZ7QMcYDCMCwBiu1tW2SFe02EJy1MxlacYIY95pLKgu-rC5OZbHBjIQzH2dnWh9FuWIIv91BLhqw-JNZ4CMGVrZb62mlyEU4pA5zS4J8NgEI_2W55mxgICbBNwg0ncaLJayXDBBWBL92b5FDF3jGJQL7MubfZwOOD3CL0nTA6JLSK81soWR2m5QPVRsPlNyDqkdVRw6yo10vfbQjfn6VtEW6u3JAkZb7VVcPO6ceUv4-VwZfDgi7_WGEcX1o6CG3DSXAW4eoj0oZjuyGjStgUZ2s--qN_D5u9i3FQsp7UWACrjMch4jpvYCYNgNkx4easPQomXURT0vd7-qXIbxhqoKQndtVZpehrnmvjdd8jdofv7XhOn-XoGZ_47rClP1tW-b7EYU-HWL-h_JLIe6Dp0bCKIO4pcK8qDCbYEE3OC2TDKDnRzQbxdyL7ubhTppUu20V4J8-pu4YMD5WKJyxkSiQ5lTRe9lAuPZ-ownQ1AOTnNu2cwmlr-6_OXk4pu5ySlXLqDfadHUPp3y-wQyz1CDjPTIfExyMGx61AaOGPkjUzACFUy5TG9DBciaqKK7TkPTVjWlanJ3N4eiBcfAkyo1lOk3rZy4sMwI6QlVDG_j_TG3cwdPd_upTjmYMa_SswYDUrI4alP_93yPG0EjfERKhFvgaD2wANq6xsvbG3gF3jWFsR4jxI8ble1ZT_jHykeun1ow4sQiop27eSCLHjb0xN8cSLXaYx4y8Rvos1XPeFpHbhsHJCTp8MgHoN1f3YI5CBLUK9QK8dAYxiai6oEnzwMJ_gHW3xRmpUm_kOwsU-cCnME1K0PfWwyjrHLUJS_u6hqrIXwY6fWXTqVJkOLudWBz3C03QHUYk331mayewXs0gpB_rNGotRQNJVtYwYntZvjQU39VwvBoXJ9ByLAZx1nShIo-rINM03CoSARe6TMKhb_T5NWubYQyYhAaLrjCv7rwNk_mDM31XXU0wzLkPMtuzADhF0QIgOuLgAAAAA%3D%3D
```

访问url时，就对上面的操作分别做一次逆操作，然后再送入vae解码，就能还原出这张图片:

![Lenna_out.webp](example/Lenna_out.webp)

看起来有一点微小的差别，不过大部分是一样的啦！


## 使用方法

首先你需要一个Python3，安装依赖:

```sh
pip install -r requirements.txt
```

启动服务: 

```sh
uvicorn main:app --host 0.0.0.0 --reload
```

然后就可以去`http://localhost:8000/`上传图片啦！


## 注意

emmmbedding看起来很像embedding，但它其实不是，它是恶魔妹妹床上用品！


## 结束

就这样，大家88，我要回去和`emmm`亲热了！
