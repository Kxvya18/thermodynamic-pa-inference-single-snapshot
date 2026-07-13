from shared.ba_experiments_full import fig1_boltzmann, fig2_arrow_of_time, fig4_identities


def main():
    out_dir = "outputs/generated"
    fig1_boltzmann(out_dir)
    fig2_arrow_of_time(out_dir)
    fig4_identities(out_dir)


if __name__ == "__main__":
    main()
